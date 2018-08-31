import os
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)

def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(name, param)

def print_nonzeros(model):
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        print(f'{name:20} | nonzeroes = {nz_count:7} / {total_params:7} | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
