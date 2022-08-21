"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pdb
import numpy as np


class CosNorm_Classifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048, scale=16.0, *args):
        super(CosNorm_Classifier, self).__init__()
        self.in_channels = feat_dim
        self.num_classes = num_classes
        self.scale = scale
        
        self.weight = Parameter(torch.Tensor(num_classes, feat_dim).cuda())
        self.reset_parameters() 
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        ex = F.normalize(input, p=2, dim=1)
        ew = F.normalize(self.weight, p=2, dim=1)
        return torch.mm(ex, self.scale * ew.t())
    
def create_model(dataset, feat_dim, num_classes=1000, scale=None, test=False, *args):
    print('Loading Cos Norm Classifier.')
    clf = CosNorm_Classifier(num_classes, feat_dim, scale, *args)

    return clf
