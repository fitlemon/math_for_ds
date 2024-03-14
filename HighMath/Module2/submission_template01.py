import numpy as np
import torch
from torch import nn

def create_model():
    # your code here
    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 16), nn.ReLU(), nn.Linear(16, 10))
    # return model instance (None is just a placeholder)
    
    return model

def count_parameters(model):
    # your code here
    # return integer number (None is just a placeholder)    
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    