import torch.nn as nn
import os
import torch
from ..core.primitives import Zero, Identity


os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False
torch.cuda.is_available = lambda : False


def _get_dense_ops(sizes):
    ops = []
    for i in sizes:
        ops.append(nn.LazyLinear(i))
        

def _get_zero_op():
    return Zero()

def _get_identity_op():
    return Identity()
