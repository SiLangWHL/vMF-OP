import torch
import torch.nn as nn
import torch.nn.functional as F
import mpmath as mp
import numpy as np
import math

def read_ni():
    mask = 1 - torch.FloatTensor( torch.eye(1000) )
    return mask.cuda()
np_iv = np.frompyfunc(mp.besseli, 2, 1) 
np_log = np.frompyfunc(mp.log, 1, 1)
np_power = np.frompyfunc(mp.power, 2, 1)

class Function_Apk(torch.autograd.Function):
    @staticmethod
    def forward(self, p, kappa): 
        kappa_cpu = kappa.data.cpu().numpy()
        v = p // 2 - 1
        self.p = p
        apk_lambda = lambda t: float(np_iv(v+1, t) / np_iv(v, t ))
        apk = np.array(list(map(apk_lambda, kappa_cpu)))
        apk = torch.from_numpy(apk).type_as(kappa).to(kappa.device)
        self.save_for_backward(kappa, apk)
        return apk
    @staticmethod
    def backward(self, grad_output):
        kappa, apk = self.saved_tensors
        grad = 1 - (self.p - 1) / kappa * apk - apk ** 2
        return None, grad_output * grad

class Function_Bias(torch.autograd.Function):
    @staticmethod
    def forward(self, p, kappa, apk): # computing I_v(z)
        self.save_for_backward(apk)
        kappa_cpu = kappa.data.cpu().numpy()
        v = p // 2 - 1
        func_lambda = lambda t: float( v * np_log(t / 2) - np_log(np_iv(v, t)) - 
                                       v * np_log(16 / 2) + np_log(np_iv(v, 16)) )
        res = np.array(list(map(func_lambda, kappa_cpu)))
        res = torch.from_numpy(res).type_as(kappa).to(kappa.device)
        return res 
    @staticmethod
    def backward(self, grad_output):
        apk = self.saved_tensors[-1]
        return None, - grad_output * apk, None

Apk = Function_Apk.apply
Bias = Function_Bias.apply

def onehot(label, num_classes):
    lbl = label.clone()
    N = num_classes + 1
    size = list(lbl.size())
    lbl = lbl.view(-1)
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, lbl.long())
    size.append(N)
    return ones.view(*size)[...,:(N-1)].float()


def get_mu(weight):
    return F.normalize(weight, p=2, dim=1)
def get_sim(mu):  #. C X D
    return torch.matmul(mu, mu.T)  ##
def get_apk(p, kappa):
    apk = Apk(p, kappa) # A_d(k) in the paper
    return apk
def get_bias(p, kappa, apk): # log(C_d(k)) in the paper
    bias = Bias(p, kappa, apk.detach())
    return bias

def get_overlap(apk, bias, kappa, sim):
    kl = apk.unsqueeze(1) * ( kappa.unsqueeze(1) - kappa.unsqueeze(0)*sim ) \
       + bias.unsqueeze(1) - bias.unsqueeze(0)
    op = 1 / ( 1 + kl )
    return op
def get_mu_feat(gt, feat, num_classes):
    feat = F.normalize(feat, p=2, dim=1)
    gt = onehot( gt.float().detach(), num_classes )
    weight = torch.matmul( gt.T, feat ) 
    count = gt.sum(0)
    mask = count >= 1
    mu_feat = weight[mask]
    mu_feat = F.normalize(mu_feat, p=2, dim=1)
    return mu_feat, mask

def get_all_overlap(op_opt, apk, bias, kappa, feat, gt, num_classes, mu, ni):
    
    mu_feat, mask = get_mu_feat(gt, feat, num_classes)
    
    sim_class = get_sim(mu)
    if 'icd' in op_opt['auxloss']:
        Op = get_overlap(apk, bias, kappa, sim_class)
        valid_op = Op #* ( 1 - Eye )
        loss_icd = ( ni * valid_op ).mean().mean() #.sum().sum() / ni.sum().sum()
    else:
        loss_icd = 0
    with torch.no_grad():
        sim_c = (ni * sim_class).mean(1).mean(0)
        op = valid_op.mean(1)
    return loss_icd, op, sim_c

ni = read_ni()
def multi_get_loss(op_opt, p, kappa, weight, feat, gt, num_classes):
    mu = get_mu(weight)
    apk = get_apk(p, kappa)
    bias = get_bias(p, kappa, apk)
    if op_opt['auxloss'] != []:
        loss_icd, op, sim_c = get_all_overlap(op_opt, apk, bias, kappa, feat, gt, num_classes, mu, ni)
    else:
        loss_icd, sim_c, op = 0.0, 0.0, bias.detach()
    return loss_icd, bias, op, sim_c

