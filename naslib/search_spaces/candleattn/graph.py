import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import *

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_str,
    convert_op_indices_to_str,
)
from naslib.search_spaces.nasbench201.encodings import encode_201, encode_adjacency_one_hot_op_indices
from naslib.utils.encodings import EncodingType

from .primitives import ResNetBasicblock

NUM_EDGES = 6
NUM_OPS = 5

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]


class CandleAttnSearchSpace(Graph):

    QUERYABLE = True

    def __init__(self, n_classes=2, num_layers=10, rel_size='large'):
        super().__init__()
        self.num_classes = n_classes
        self.rel_size = rel_size
        self.num_layers = num_layers
        self.space_name = "candleattn"
        self.instantiate_model = True
        self.sample_without_replacement = False

        #
        # Cell definition
        #

        self.name = "makrograph"
        anchor_points = collections.deque([source], maxlen=3)
        # preprocessing
        self.edges[1, 2].set("op", ops.Stem(C_in=self.in_channels,
                                            C_out=self.channels[0]))
        for i in range(2, total_num_nodes + 2):
            self.add_node(k)
            self.add_node(k+1)
            self.add_edge(k, k+1)
            #TODO: add dense
            self.edges[k, k+1].data.set("op", ops._get_dense_ops([2000, 2001, 2002, 2003]))
            self.add_node(k+2)
            self.add_edge(k+1, k+2)
            self.edges[k+1, k+2].set("op", ops._get_identity_op())
            k+=3
            old_k = k-1
            for anchor in anchor_points:
                self.add_node(k)
                self.add_edge(k-1, k)
                self.edges[k-1, k].set("op", [ops._get_zero_op(), ops._get_identity_op()]) 
                self.add_edge(k, old_k-1)             
            prev_input = old_k-1
            anchor_points.append(prev_input)

    

    def get_op_indices(self) -> list:
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def get_hash(self) -> tuple:
        return tuple(self.get_op_indices())

    def get_arch_iterator(self, dataset_api=None) -> Iterator:
        return itertools.product(range(NUM_OPS), repeat=NUM_EDGES)

    def set_op_indices(self, op_indices: list) -> None:
        self.op_indices = op_indices

    def set_spec(self, op_indices: list, dataset_api=None) -> None:
        self.set_op_indices(op_indices)

    def sample_random_labeled_architecture(self) -> None:
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        op_indices = random.choice(self.labeled_archs)

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api: dict = None, load_labeled: bool = False) -> None:
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        def is_valid_arch(op_indices: list) -> bool:
            return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or
                        (op_indices[2] == op_indices[4] == op_indices[5] == 1))

        while True:
            op_indices = np.random.randint(NUM_OPS, size=(NUM_EDGES)).tolist()

            if not is_valid_arch(op_indices):
                continue

            self.set_op_indices(op_indices)
            break
        self.compact = self.get_op_indices()

    def mutate(self, parent: Graph, dataset_api: dict = None) -> None:
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = parent.get_op_indices()
        op_indices = list(parent_op_indices)

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
        op_index = np.random.choice(available)
        op_indices[edge] = op_index
        self.set_op_indices(op_indices)

    def mutate_with_parent_op_indices(self, parent_op_indices: typing.Any, dataset_api: dict = None) -> None:
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        op_indices = list(parent_op_indices)

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
        op_index = np.random.choice(available)
        op_indices[edge] = op_index
        self.set_op_indices(op_indices)


    def get_type(self) -> str:
        return "candleattn"

    def get_loss_fn(self) -> Callable:
        return F.cross_entropy

    '''
    def forward_before_global_avg_pool(self, x: torch.Tensor) -> list:
        outputs = []

        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
        return encode_201(self, encoding_type=encoding_type)

    '''
