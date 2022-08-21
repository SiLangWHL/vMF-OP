import mpmath as mp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.functional import interpolate
#from utils import Function_Bias, Function_Apk
import mpmath as mp
import numpy as np
from scipy.special import iv
np_iv = np.frompyfunc(mp.besseli, 2, 1)
np_log = np.frompyfunc(mp.log, 1, 1)
def read_ni(dir="./n_i_pre.npy"):
    ni = torch.FloatTensor( np.load(dir) )
    cls_weight = torch.log( ni )
    return cls_weight

def get_ni(weight):
    ni = read_ni().type_as(weight).to(weight.device)
    return ni

class Multi_kappa_vMF_Classifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048, k_init = 16.0, alpha = 1.0, *args):
        super(Multi_kappa_vMF_Classifier, self).__init__()
        self.in_channels = self.p = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim), requires_grad=True)
        self.register_buffer('Bias', get_ni(self.weight).unsqueeze(0) ) # this is the training_set prior distribution, 'bias' 
        self.register_buffer('bias', torch.zeros(self.num_classes) ) # this is the training_set prior distribution, 'bias' 
        self.register_buffer('op', torch.zeros(self.num_classes) )
        self.register_buffer('ni', torch.zeros(self.num_classes) )
        self.reset_parameters(self.weight)
        self.Kappa = torch.nn.Parameter( torch.FloatTensor( [k_init]*self.num_classes  ) )
    def post_process(self):
        op_min, op_max = self.op.min(), self.op.max()
        kappa_min, kappa_max = self.kappa.min(), self.kappa.max()
        op_reset = (self.op - op_min) / (op_max - op_min) * (kappa_max - kappa_min) + kappa_min
        kappa_reset = self.kappa ** self.alpha * op_reset ** (1-self.alpha)
        return kappa_reset
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
    def _get_mu_kappa(self):
        self.mu = F.normalize(self.weight, p=2, dim=1)
        self.kappa = torch.clamp(self.Kappa, 1.0)
    def _get_all_var(self):
        dic = {}
        dic['weight'] = self.weight
        dic['kappa'] = self.Kappa
        dic['p'] = self.p
        dic['num_classes'] = self.num_classes
        return dic 
    def forward(self, x, bias=None):
        self._get_mu_kappa()
        x_norm = F.normalize(x, p=2, dim=1) 
        if not self.training :
            if self.alpha is not None:
                kappa = self.post_process()
            else:
                kappa = self.kappa
            score = torch.mm( x_norm, ( self.mu * kappa.unsqueeze(1) ).t() ) + self.bias.unsqueeze(0)
            return score
        score = torch.mm( x_norm, ( self.mu * self.kappa.unsqueeze(1) ).t() ) + self.Bias + bias.unsqueeze(0)
        return score


def create_model(dataset, feat_dim=64, num_classes=100, alpha = 1.0, k_init=None, test=False, *args):
    print('Loading vMF Classifier with k_init: {}.'.format(k_init))
    
    clf = Multi_kappa_vMF_Classifier(num_classes, feat_dim, k_init, alpha, *args)

    return clf
