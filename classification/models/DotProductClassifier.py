import torch.nn as nn
import numpy as np
from utils import *
from os import path

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)
    def forward(self, x):
        x = self.fc(x)
        return x
    
def create_model(dataset, feat_dim, num_classes=1000, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim, *args)
    return clf
