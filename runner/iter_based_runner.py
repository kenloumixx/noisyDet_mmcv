# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import sys
import platform
import shutil
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, no_type_check

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel


import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.optim as optim

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .hooks import IterTimerHook
from .utils import get_host_info
from .splitnet import SplitNet
from .dist_utils import get_dist_info, allreduce_params
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn as nn
import os
from multiprocessing import Pool
import time 
import threading
# from threading import Event
import re 
import json

class IterLoader:

    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self._epoch = 0
        # self.iter_loader = iter(self._dataloader)

    @property
    def epoch(self) -> int:
        return self._epoch

    def make_iter(self):
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            rank, world_size = get_dist_info()  
            if rank == 0:
                print(f'\n epoch plus tobe {self._epoch}!\n')
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)

def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

def all_gather(tensor: torch.Tensor, fixed_shape: Optional[List] = None) -> List[torch.Tensor]:
    def compute_padding(shape, new_shape):
        padding = []
        for dim, new_dim in zip(shape, new_shape):
            padding.insert(0, new_dim - dim)
            padding.insert(0, 0)
        return padding

    input_shape = tensor.shape
    if fixed_shape is not None:
        padding = compute_padding(tensor.shape, fixed_shape)
        if sum(padding) > 0:
            tensor = F.pad(tensor, pad=padding, mode='constant', value=0)
    
    output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)

    all_input_shapes = None
    if fixed_shape is not None:
        # gather all shapes
        tensor_shape = torch.tensor(input_shape, device=tensor.device)
        all_input_shapes = [torch.zeros_like(tensor_shape) for _ in range(dist.get_world_size())]
        dist.all_gather(all_input_shapes, tensor_shape)
        all_input_shapes = [t.tolist() for t in all_input_shapes]

    if all_input_shapes:
        for i, shape in enumerate(all_input_shapes):
            padding = compute_padding(output[i].shape, shape)
            if sum(padding) < 0:
                output[i] = F.pad(output[i], pad=padding)

    return output


def all_gather_object(object_list, obj, group=None):
    import contextlib
    import io
    import logging
    import os
    import pickle
    import time
    import warnings
    from datetime import timedelta
    from typing import Callable, Dict, Optional, Tuple, Union
    
    _pickler = pickle.Pickler
    _unpickler = pickle.Unpickler
    
    def _tensor_to_object(tensor, tensor_size):
        buf = tensor.numpy().tobytes()[:tensor_size]
        return _unpickler(io.BytesIO(buf)).load()

    def _object_to_tensor(obj):
        f = io.BytesIO()
        _pickler(f).dump(obj)
        byte_storage = torch.ByteStorage.from_buffer(f.getvalue())  # type: ignore[attr-defined]
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will casue 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage)
        local_size = torch.LongTensor([byte_tensor.numel()])
        return byte_tensor, local_size


    input_tensor, local_size = _object_to_tensor(obj)
    current_device = torch.device("cuda", torch.cuda.current_device())
    input_tensor = input_tensor.to(current_device)
    local_size = local_size.to(current_device)

    group_size = torch.distributed.get_world_size()
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    group = torch.distributed.new_group(list(range(group_size)))
    torch.distributed.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device
    )

    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, input_tensor, group=group)

    k = 0
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device("cpu"):
            tensor = tensor.cpu()

        tensor_size = object_size_list[i]
        output = _tensor_to_object(tensor, tensor_size)
        for j in range(len(output)):
            object_list[k] = output[j]
            k += 1    


@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    def train(self, data_loader, rank, **kwargs):
        rank, world_size = get_dist_info()
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader[0]
        self._epoch = self.data_loader.epoch
        data_batch = next(self.data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        # 2. ssod/models/soft_teacher.py forward_train
        # 1. mmdet/models/detectors/base.py -> train_step
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1


    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_val_iter')
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        del self.data_batch
        self._inner_iter += 1

    '''
    def insert_CN_label_splitnet(self, clean_noise_label, dataset, box_ids):    # dataset??? training??? ????????? ??????
        for idx, label in enumerate(clean_noise_label): # ????????? ???????????????
            dataset[idx]['gmm_labels'] = label.item()
            dataset[idx]['box_ids'] = label.item()
    '''

    def insert_CN_label(self, clean_noise_label, bbox_ids, GMM_GT_idx, dataset):    # dataset??? training??? ????????? ??????
        for bbox_idx, label, gmm_gt in zip(bbox_ids, clean_noise_label, GMM_GT_idx): # ????????? ???????????????
            dataset.sup.coco.anns[bbox_idx.item()]['gmm_labels'] = label
            dataset.unsup.coco.anns[bbox_idx.item()]['gmm_labels'] = label

            dataset.sup.coco.anns[bbox_idx.item()]['GMM_GT_idx'] = gmm_gt.item()
            dataset.unsup.coco.anns[bbox_idx.item()]['GMM_GT_idx'] = gmm_gt.item()
        
    def train_splitnet_1epoch(self, epoch, dataloader, len_dataset):
        # self.ddp_splitnet.train()
        self.splitnet.train()
        rank, world_size = get_dist_info()
        prog_bar = mmcv.ProgressBar(len_dataset)
        
        for batch_idx, data in enumerate(dataloader): # TODO ??? ????????? cuda:0?????? ??????????????????.. -> data loader - sampler????????? dist????????? cuda rank??? ?????????. ????????? data??? loader??? ????????? ?????? ????????? cpu??? ???????????? ???!
            # if batch_idx >= len(dataloader):
            #     break
            logits, noisy_label, loss_bbox, gmm_labels, logits_delta, loss_bbox_delta = data['logits'], data['cls_labels'], data['loss_bbox'], data['gmm_labels'], data['logits_delta'], data['loss_bbox_delta']
            logits, noisy_label, loss_bbox, gmm_labels, logits_delta, loss_bbox_delta = logits.to(rank), noisy_label.to(rank), loss_bbox.to(rank), gmm_labels.to(rank), logits_delta.to(rank), loss_bbox_delta.to(rank)
            self.optimizer_splitnet.zero_grad()

            predict = self.splitnet(logits=logits, noisy_label=noisy_label, loss_bbox=loss_bbox, logits_delta=logits, loss_bbox_delta=loss_bbox, epoch=self.epoch)  # ????????? ???????????????..??????

            loss = self.splitnet_loss_func(predict, gmm_labels)

            if loss < 0.001:
                print(f'\n{self._iter} | loss is less than {0.001}\n')
                break

            # self.splitnet_scaler.scale(loss).backward() 
            loss.backward()

            # self.splitnet_scaler.step(self.optimizer_splitnet)
            self.optimizer_splitnet.step()
            # self.splitnet_scaler.update()
            for _ in range(logits.shape[0]):
                prog_bar.update()
        print(f'\niter {self._iter} training finish!\n')
            
    def total_tensor(self, tensor, max_num):      # 1. ?????? 2. ??????????????? 3. ?????? ????????? -> max num ??? ??? ???????????? ?????? ???????????????..!
        rank, world_size = get_dist_info()
        # 1. padding?????? 
        delta = max_num - tensor.size(0)            # batch ????????? ?????? ????????? tensor size ????????????
        # delta?????? concat
        tensor = torch.cat([tensor, torch.zeros_like(tensor)[:delta]])   # ?????? ?????? ?????????
        
        # 2. ???????????????
        output = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor) # -> output = [tensor, tensor, ..., tensor]
        
        # 3. delta?????? ????????????
        total_delta = [None for _ in range(world_size)]
        rank_delta = [delta]    # ????????? ?????? ????????? ????????????
        all_gather_object(total_delta, rank_delta)       # total_delta [309, 158, 276, 185, 0, 252, 323, 635]

        # 4. ?????? ???????????? -> ?????? ???????????? ?????????, ???????????? pad??? ????????? ?????? ?????? concat??????
        total_output = torch.cat(output)    # output stack?????? ??? ????????????!   # total_output torch.Size([8, 4809, 81]) -> 38472
        chunked_output_list = list(torch.tensor_split(total_output, world_size))


        for idx, rank_delta in enumerate(total_delta):
            if rank_delta == 0:
                continue
            chunked_output_list[idx] = chunked_output_list[idx][:-rank_delta]

        output = torch.cat(chunked_output_list)
        return output


    @torch.no_grad()
    def val_splitnet(self, dataloader, len_dataset):
        self.splitnet.eval()
        rank, world_size = get_dist_info()
        predict_total = []
        prob_total = []
        maxidx_total = []
        box_ids_total = []

        prog_bar = mmcv.ProgressBar(len_dataset)

        for batch_idx, data in enumerate(dataloader):
            # if batch_idx >= len(dataloader):
            #     break
            logits, noisy_label, loss_bbox, box_ids, logits_delta, loss_bbox_delta = data['logits'], data['cls_labels'], data['loss_bbox'], data['box_ids'], data['logits_delta'], data['loss_bbox_delta']
            logits, noisy_label, loss_bbox, box_ids, logits_delta, loss_bbox_delta = logits.to(rank), noisy_label.to(rank), loss_bbox.to(rank), box_ids.to(rank), logits_delta.to(rank), loss_bbox_delta.to(rank)
            
            # with torch.cuda.amp.autocast():
            with torch.no_grad():
                predict = self.splitnet(logits=logits, noisy_label=noisy_label, loss_bbox=loss_bbox, logits_delta=logits_delta, loss_bbox_delta=loss_bbox_delta, epoch=self.epoch)

            predict = torch.softmax(predict, dim=-1)
            prob, maxidx = torch.max(predict, dim=-1)  

            predict_total.extend(predict)
            prob_total.extend(prob)
            maxidx_total.extend(maxidx)
            box_ids_total.extend(box_ids)
            
            for _ in range(logits.shape[0]):
                prog_bar.update()



        '''    
        # ????????? ?????????..? <- ?????? ????????? ?????? ??? ????????? ?????????.. ?????? ???????????????
        rank, world_size = get_dist_info()
        device = torch.device(f'cuda:{rank}')
        
        # ??? bbox?????? ?????? ?????? ?????????
        num_total_data = [None for _ in range(world_size)]
        data_len = [len(predict_total)]
        all_gather_object(num_total_data, data_len)
        num_bbox = sum(num_total_data)  # total bbox len
        max_num = max(num_total_data)  # total bbox len

        predict_total_list = torch.stack(predict_total)
        maxidx_total_list = torch.stack(maxidx_total)
        box_ids_total_list = torch.stack(box_ids_total)

        predict_total_tensor = total_tensor(predict_total_list, max_num)
        maxidx_total_tensor = total_tensor(maxidx_total_list, max_num)
        box_ids_total_tensor = total_tensor(box_ids_total_list, max_num)
        '''
    

        ''' latest for binary - 0107
        # 1. ????????? ????????? 
        predict_total_tensor = torch.stack(predict_total)   # ,4
        maxidx_total_tensor = torch.stack(maxidx_total)
        box_ids_total_tensor = torch.stack(box_ids_total).to(rank)

        num_total_data = [None for _ in range(world_size)]
        data_len = [len(predict_total)]
        all_gather_object(num_total_data, data_len)
        max_num = max(num_total_data)  

        total_predict_total_tensor = self.total_tensor(predict_total_tensor, max_num)
        total_maxidx_total_tensor = self.total_tensor(maxidx_total_tensor, max_num)
        total_box_ids_total_tensor = self.total_tensor(box_ids_total_tensor, max_num)

        # 2. ????????? clean_noise_label ????????? 
            
        thres = 0.5 # TODO
        
        predict_thre = total_predict_total_tensor.ge(thres).detach().cpu().numpy()
        maxidx_total_thre = total_maxidx_total_tensor.eq(0).detach().cpu().numpy()

        # total_clean_noise_label <- 4?????? classes?????? ???
        total_clean_noise_label = predict_thre * maxidx_total_thre      # ??? ?????? ????????? ??????????????????        
        
        # 3. rank??? ?????? ?????? ??????????????? 
        rank_partition = len(total_predict_total_tensor) // world_size * rank
        rank_partition_plus = len(total_predict_total_tensor) // world_size * (rank + 1)

        if rank == world_size - 1:        
            clean_noise_label = total_clean_noise_label[rank_partition : rank_partition_plus]
            box_ids_total_tensor = total_box_ids_total_tensor[rank_partition : rank_partition_plus]
        else:
            clean_noise_label = total_clean_noise_label[rank_partition :]
            box_ids_total_tensor = total_box_ids_total_tensor[rank_partition :]            
        
        return clean_noise_label, box_ids_total_tensor
        '''

        # 112 start
        # 1. ????????? ????????? 
        total_predict_total_tensor = torch.stack(predict_total).to(rank)
        total_box_ids_total_tensor = torch.stack(box_ids_total).to(rank)

        # num_total_data = [None for _ in range(world_size)]

        # data_len = [len(predict_total)]

        # all_gather_object(num_total_data, data_len)

        # max_num = max(num_total_data)  
        
        # total_predict_total_tensor = self.total_tensor(predict_total_tensor, max_num)
        # total_box_ids_total_tensor = self.total_tensor(box_ids_total_tensor, max_num)
        total_clean_noise_label = total_predict_total_tensor.detach().cpu().numpy()      # ??? ?????? ????????? ??????????????????        
        # 112 fin

        '''
        # 2. ????????? clean_noise_label ????????? 
        # total_clean_noise_label <- 4?????? classes?????? ???
        total_clean_noise_label = total_predict_total_tensor.detach().cpu().numpy()      # ??? ?????? ????????? ??????????????????        
        
        # 3. rank??? ?????? ?????? ??????????????? 
        rank_partition = len(total_predict_total_tensor) // world_size * rank
        rank_partition_plus = len(total_predict_total_tensor) // world_size * (rank + 1)

        if rank == world_size - 1:        
            clean_noise_label = total_clean_noise_label[rank_partition : rank_partition_plus]
            box_ids_total_tensor = total_box_ids_total_tensor[rank_partition : rank_partition_plus]
        else:
            clean_noise_label = total_clean_noise_label[rank_partition :]
            box_ids_total_tensor = total_box_ids_total_tensor[rank_partition :]            

        return clean_noise_label, box_ids_total_tensor
        '''
        del predict_total, prob_total, maxidx_total, box_ids_total
        return total_clean_noise_label, total_box_ids_total_tensor
    
    # self.??? ?????????
    def splitnet_process(self, splitnet_data, build_dataset, build_dataloader, mmdet_dataloader, cfg, splitnet_data_train_idx, dataset, GMM_GT_idx):    # ?????? ????????? ????????? ?????? ??????????????????..!
        rank, world_size = get_dist_info()
        # self.event.clear()
        self.merge_value = 0
        self.event_set_list = [0 for _ in range(world_size)]

        if rank == 0:
            self.splitnet = SplitNet(80, use_delta_logit=False).to(rank)   # cuda:0 is used to splitnet     # ?????? ?????? ???????????? training
            self.optimizer_splitnet = optim.AdamW(self.splitnet.parameters(), weight_decay=0.0005,
                                lr=0.002)  # TODO
            self.splitnet_loss_func = nn.CrossEntropyLoss()
            print(f'\nsplitnet process start {self._iter} | \n')
            distributed = False
            # ????????? splitnet_train_data ?????????

            splitnet_train_data = [data[splitnet_data_train_idx] for data in splitnet_data]
            cfg.data.gmm_coco.splitnet_data = splitnet_train_data # ????????? ???????????? ????????? ????????? ???????????? ????????? ??????..?         
            cfg.data.gmm_coco.test_mode = False
            splitnet_train_dataset = build_dataset(cfg.data.gmm_coco)
            len_splitnet_train_dataset = len(splitnet_train_dataset)

            splitnet_dataloader = mmdet_dataloader(
                splitnet_train_dataset,
                cfg.data.gmm_coco.samples_per_gpu,
                cfg.data.gmm_coco.workers_per_gpu,  # let samples_per_gpu and workers_per_gpu be the same
                # cfg.gpus will be ignored if distributed
                dist=distributed,
                seed=cfg.seed,
                runner_type='EpochBasedRunner',
                shuffle=True,
            )
            # training option??? COCO??? ????????????
            for epoch in range(1):
                self.train_splitnet_1epoch(epoch, splitnet_dataloader, len_splitnet_train_dataset)

            cfg.data.gmm_coco.splitnet_data = splitnet_data # ????????? ???????????? ????????? ????????? ???????????? ????????? ??????..?         
            splitnet_dataset = build_dataset(cfg.data.gmm_coco)
            len_splitnet_dataset = len(splitnet_dataset)
            
            # val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
            splitnet_val_dataloader = build_dataloader(
                splitnet_dataset,
                samples_per_gpu=cfg.data.gmm_coco.samples_per_gpu,
                workers_per_gpu=cfg.data.gmm_coco.workers_per_gpu,
                dist=distributed,
                shuffle=False,
            )

            splitnet_clean_noise_label, splitnet_box_ids = self.val_splitnet(splitnet_val_dataloader, len_splitnet_dataset)       # ?????? ???????????? ??????!

            self.insert_CN_label(splitnet_clean_noise_label, splitnet_box_ids, GMM_GT_idx, dataset)
            del splitnet_clean_noise_label, splitnet_box_ids, GMM_GT_idx
            del splitnet_dataloader, splitnet_val_dataloader, splitnet_dataset, splitnet_train_data


            # ?????? gpu??? event_value = 1??? ?????????
            # print(f'\n\nrank {rank} set !\n\n')
            # self.event.set()
            # all_gather_object(self.event_set_list, [int(self.event.is_set())])
            self.merge_value = 1
            all_gather_object(self.event_set_list, [self.merge_value])
            # ?????? ?????? ?????????
        else : 
            while True: 
                # all_gather_object(self.event_set_list, [int(self.event.is_set())])
                all_gather_object(self.event_set_list, [self.merge_value])
                if self.event_set_list[0] > 0:
                    break
            # self.event.set()
        
        dist.barrier()
        # dataloader.make_iter()
        
        
    def run(self,
            workflow, 
            cfg,
            distributed,
            dataset,
            build_dataloader, 
            mmdet_dataloader, 
            build_dataset,
            flag=None,
            max_iters: Optional[int] = None,
            **kwargs) -> None:
        # """Start running.

        # Args:
        #     data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
        #         and validation.
        #     workflow (list[tuple]): A list of (phase, iters) to specify the
        #         running order and iterations. E.g, [('train', 10000),
        #         ('val', 1000)] means running 10000 iterations for training and
        #         1000 iterations for validation, iteratively.
        # """
        # temp 
        rank, world_size = get_dist_info()
        
        # for splitnet
        num_cls = dataset[0].datasets[1].CLASSES    # coco class??? ??????????????? ????????? ??????
        self.splitnet_batch_size = 128  # TODO  # 32, 64, 128

        assert mmcv.is_list_of(workflow, tuple)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')
        
        # ??? ????????? ?????? ????????? loader??? ????????? ??????????????????
        # train????????? ??? iter_loaders??? ???????????? ??????
        # gmm_epoch ?????? ????????? iter_loaders ??????
        self.call_hook('before_epoch')  # self.gmm_loader 
        self.event = threading.Event()
        # for trainig dataloader 
        data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed,
                sampler_cfg=cfg.data.get("sampler", {}).get("train", {}),
            )
            for ds in dataset
        ]   

        sampler = cfg.data.get("sampler", {}).get("train", {})
        self.start = True

        assert isinstance(data_loaders, list)
        assert len(data_loaders) == len(workflow)

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.bbox_check = open('save.txt', 'w')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                iter_runner = getattr(self, mode)       # iterbasedrunner.train
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    # ??????!
                    if ((self.iter+1) % 7350 == 0) or self.start:
                        # print(f'self.iter {self.iter}') # 2000?????? ???????????? ???????????? ??? self.iter??? ????????? ?????????????    # ?????? 1999?????? ???
                        # exit()
                        if rank == 0:
                            print(f'rank {rank} | self._iter {self.iter} | self._iter {self._iter} | self._inner_iter {self._inner_iter}')
                        # gmm_dataset??? train dataset ???????????? ?????????.. ??? ????????? sup???..   -> ??? ???????????? gmm??? training dataset??? ???????????? ???????
                        splitnet_data, splitnet_data_train_idx, GMM_GT_idx = self.call_hook('gmm_epoch')  # ????????? ???????????? gmm??? label -> (0~3)                        

                        if (self.iter+1) // 7350 == 0:  # ???????????? ???????????? ????????? ????????? ????????? 
                            logits_delta = splitnet_data[2]
                            loss_bbox_delta = splitnet_data[1]

                            # prev_logits = splitnet_data[2]
                            # prev_loss_bbox = splitnet_data[1]

                        else:   # ?????????????????? delta??? ???????????? ???. # self.start?????? ????????? 7350??? 2????????? ???????????? 
                            if self.start:  # ?????? ???????????? ????????? ???????????? ?????????????????????
                                if os.path.exists("splitnet_delta.json"):
                                    with open("splitnet_delta.json", "r") as json_file: 
                                        splitnet_delta = json.load(json_file)
                                        if rank == 0:
                                            print('\nsplitnet json file loaded! \n')
                                    prev_logits = torch.tensor(splitnet_delta['prev_logits']).to(rank)
                                    prev_loss_bbox = torch.tensor(splitnet_delta['prev_loss_bbox']).to(rank)

                            logits_delta = splitnet_data[2] - prev_logits
                            loss_bbox_delta = splitnet_data[1] - prev_loss_bbox
                        
                        splitnet_data.append(logits_delta)
                        splitnet_data.append(loss_bbox_delta)

                        prev_logits = splitnet_data[2]  # ????????? ?????? ?????????..? -> ?????????????????? ????????????
                        prev_loss_bbox = splitnet_data[1]

                            
                        if rank == 0:
                            if ((self.iter+1) % 7350 == 0) or ((self.iter+1) == 2000):     # ????????? ?????? ????????? ?????? ??????.. ????????? ?????? load?????? ????????? ????????? ?????????..
                                # with open("history.json", "w") as json_file: 
                                with open("splitnet_delta.json", "w") as json_file: 
                                    json.dump({'iter': self.iter+1, 'prev_logits': splitnet_data[2].cpu().numpy().tolist(), 'prev_loss_bbox': splitnet_data[1].cpu().numpy().tolist()}, json_file)
                                    print('\nsplitnet json file dumped! \n')
                                
                        splitnet_data_cpu = [elem.cpu() for elem in splitnet_data]  # cpu??? ?????????..!
                        del splitnet_data
                        # splitnet_data_cpu.append(GMM_GT_idx)
                            
                        # ????????? splitnet ???????????????

                        '''
                        if rank == 0:
                            self.splitnet_process(splitnet_data_cpu, build_dataset, build_dataloader, mmdet_dataloader, cfg, splitnet_data_train_idx, iter_loaders[0]._dataloader.dataset)
                            
                        while True:
                            if len(os.listdir('save_dir')) == world_size:
                                break
                            elif (f'save_{rank}.txt' in os.listdir('save_dir')) == False:
                                f = open(f'save_dir/save_{rank}.txt', 'w')
                                f.close()
                            time.sleep(0.1)                                
                        '''
                        # starting thread 1
                        
                        # threading.Thread(target=self.splitnet_process, args=(splitnet_data_cpu, build_dataset, build_dataloader, mmdet_dataloader, cfg, splitnet_data_train_idx, iter_loaders[0]._dataloader.dataset)).start()
                        self.splitnet_process(splitnet_data_cpu, build_dataset, build_dataloader, mmdet_dataloader, cfg, splitnet_data_train_idx, iter_loaders[0]._dataloader.dataset, GMM_GT_idx)
                        # self.event.set()
                        # ?????? ??????????????? group?????? ?????? ????????? t2??? ???????????? ?????? ??????..
                        del splitnet_data_cpu
                    
                        # wait until thread 1 is completely executed
                        # t1.join()
                        
                        # else:
                        #     while True:
                        #         if len(os.listdir('save_dir')) == world_size:
                        #             break
                        #         else:
                        #             if f'save_{rank}.txt' in os.listdir('save_dir'):
                        #                 pass
                        #             else:
                        #                 f = open(f'save_dir/save_{rank}.txt', 'w')
                        #                 f.close()
                        #         time.sleep(0.1)
                        # ????????? rank ?????????
                        if self.start:
                            iter_loaders[0].make_iter()
                            self.start = False

                    # ???!
                    # insert????????? probability 4-dim  -> new_clean_noise_label??? 4dim?????? tensor?????? ???????????? ???
                    
                    # allreduce_params(self.ddp_splitnet.buffers())
                    # splitnet_filepath = osp.join(self.work_dir, f'splitnet_epoch_{self._epoch}.pth')
                    # save_checkpoint(self.ddp_splitnet, splitnet_filepath, optimizer=self.optimizer_splitnet)  # for checkpoint

                    # ????????? insert??? ????????? ????????? ???????????? A
                    iter_runner(iter_loaders, rank, **kwargs) # iterloader?????? ???????????? ?????? ?????????1
                    
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    def splitnet_latest_checkpoint(self, work_dir):
        path_list = os.listdir(work_dir)
        pattern = re.compile('splitnet_epoch_[0-9]+')
        splitnet_path = [path for path in path_list if pattern.match(path)]
        if len(splitnet_path) == 0:
            return None
        last_splitnet_weight = sorted(splitnet_path, key=lambda x: int(x.split('.')[0][15:]))[-1]
        return last_splitnet_weight

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               splitnet: bool = False,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                splitnet, 
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                splitnet, checkpoint, map_location=map_location)

        if splitnet == False:
            self._epoch = checkpoint['meta']['epoch']
            self._iter = checkpoint['meta']['iter']
            self._inner_iter = checkpoint['meta']['iter']
                    
        if 'optimizer' in checkpoint and resume_optimizer:
            if splitnet == False:
                if isinstance(self.optimizer, Optimizer):
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                elif isinstance(self.optimizer, dict):
                    for k in self.optimizer.keys():
                        self.optimizer[k].load_state_dict(
                            checkpoint['optimizer'][k])
                else:
                    raise TypeError(
                        'Optimizer should be dict or torch.optim.Optimizer '
                        f'but got {type(self.optimizer)}')
            else:
                if isinstance(self.optimizer_splitnet, Optimizer):
                    self.optimizer_splitnet.load_state_dict(checkpoint['optimizer'])
                elif isinstance(self.optimizer_splitnet, dict):
                    for k in self.optimizer_splitnet.keys():
                        self.optimizer_splitnet[k].load_state_dict(
                            checkpoint['optimizer'][k])
                else:
                    raise TypeError(
                        'Optimizer should be dict or torch.optim.Optimizer '
                        f'but got {type(self.optimizer_splitnet)}')
                


        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

    def save_checkpoint(  # type: ignore
            self,
            out_dir: str,
            filename_tmpl: str = 'iter_{}.pth',
            meta: Optional[Dict] = None,
            save_optimizer: bool = True,
            create_symlink: bool = True) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        
        optimizer = self.optimizer if save_optimizer else None
        # optimizer_splitnet = self.optimizer_splitnet if save_optimizer else None
        # splitnet_filepath = osp.join(self.work_dir, f'splitnet_epoch_{self._epoch}.pth')

        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # save_checkpoint(self.ddp_splitnet, splitnet_filepath, optimizer=self.optimizer_splitnet)  # for checkpoint

        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                custom_hooks_config=None):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', False)  # type: ignore
        if lr_config is not None:
            lr_config.setdefault('by_epoch', False)  # type: ignore
        if log_config is not None:
            for info in log_config['hooks']:
                info.setdefault('by_epoch', False)
        super().register_training_hooks(
            lr_config=lr_config,
            momentum_config=momentum_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=IterTimerHook(),
            custom_hooks_config=custom_hooks_config)
