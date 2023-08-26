import torch.nn as nn

#from ..core.primitives import AbstractPrimitive, ReLUConvBN
from ..core.primitives import * 

def _get_dense_ops(sizes):
    ops = []
    for i in sizes:
        ops.append(nn.LazyLinear(i))
        

def _get_constant_op():


def _get_add_project_op():

def _get_zero_op():
    return Zero()

def _get_identity_op():
    return Identity()
