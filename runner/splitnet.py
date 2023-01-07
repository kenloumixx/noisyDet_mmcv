import torch.backends.cudnn as cudnn
import torch
import torchvision.models as models
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.metrics import *
from torch.cuda.amp import autocast, GradScaler
import contextlib
from collections import Counter, OrderedDict
import torch.nn as nn

import torch.distributed as dist


def one_hot(targets, nClass):
    logits = torch.zeros(targets.size(0), nClass).cuda()
    return logits.scatter_(1, targets.unsqueeze(1), 1)
    # scatter_(): Expected dtype int64 for index. 

class Entropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return -torch.sum(probs.log() * probs, dim=1)


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class SplitNet(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed=256, use_delta_logit=False):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.use_delta_logit = use_delta_logit
        self.nb_classes = nb_classes
        # self.fc0 = nn.Linear(512, sz_embed)

        self.fc1 = nn.Linear(162, sz_embed)  #
        self.batch1 = torch.nn.BatchNorm1d(sz_embed)

        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(sz_embed, sz_embed // 2)
        self.batch2 = torch.nn.BatchNorm1d(sz_embed // 2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(sz_embed // 2, sz_embed // 2)
        self.batch3 = torch.nn.BatchNorm1d(sz_embed // 2)
        self.relu3 = nn.ReLU()
        self.fc6 = nn.Linear(sz_embed // 2, 2)

    def forward(self, delta, logits, noisy_label, loss_bbox):  # , mask, use_mask=False
        noisy_label = one_hot(noisy_label, self.nb_classes) # Expected dtype int64 for index.
        loss_bbox = loss_bbox.unsqueeze(-1)
        X = torch.cat((logits, noisy_label, loss_bbox), dim=-1)  # feature

        out_f = self.fc1(X)
        out_f = self.batch1(out_f)
        out_f = self.relu1(out_f)

        out_f = self.fc2(out_f)
        out_f = self.batch2(out_f)
        out_f = self.relu2(out_f)

        out_f = self.fc3(out_f)
        out_f = self.batch3(out_f)
        out_f = self.relu3(out_f)
        out_f = self.fc6(out_f)     # binary output

        return out_f    # 4,2
    
