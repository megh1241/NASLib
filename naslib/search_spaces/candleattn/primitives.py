import torch.nn as nn
import os
import torch
from ..core.primitives import AbstractPrimitive, Zero, Identity


#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#torch.backends.cudnn.enabled = False
#torch.cuda.is_available = lambda : False

class DenseFC(AbstractPrimitive):
    def __init__(self, size, **kwargs):
         super().__init__(locals())
         self.size = size
         self.op = nn.LazyLinear(self.size)

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None
    
    def get_op_name(self):
        return 'OPDense'+str(self.size)


def _get_dense_ops(sizes):
    ops = []
    for i in sizes:
        ops.append(_get_dense_op(i))
    return ops

def _get_dense_op(size):
    return DenseFC(size)


def _get_zero_op():
    return Zero(stride=1)

def _get_identity_op():
    return Identity()


def get_zero_op_names():
    return _get_zero_op().get_op_name()


def get_identity_op_names():
    return  _get_identity_op().get_op_name()


def get_dense_op_names(choices):
    names = []
    for ele in choices:
        names.append(_get_dense_op(ele).get_op_name())
    return names
