import collections
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import *

from naslib.search_spaces.candleattn import primitives as ops
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.candleattn.conversions import * 
from naslib.utils.encodings import EncodingType


NUM_EDGES = 6
NUM_OPS = 5

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]


class CandleAttnSearchSpace(Graph):

    QUERYABLE = True

    def __init__(self, n_classes=2, num_layers=10, rel_size='large'):
        super().__init__()
        self.num_classes = n_classes
        self.op_indices = None
        self.rel_size = rel_size
        self.num_layers = num_layers
        self.space_name = "candleattn"
        self.instantiate_model = True
        self.sample_without_replacement = False
        self.v_edges = []
        self.op_names = []
        #
        # Cell definition
        #
        self.name = "makrograph"
        # preprocessing
        self.add_node(1)
        self.add_node(2)
        self.add_edge(1, 2)
        self.edges[1, 2].set("op",  ops._get_identity_op())
        k = 3
        dense_op_list = []
        skip_op_list = []
        op_names_skip = [ops.get_zero_op_names(), ops.get_identity_op_names()]
        op_names_dense = ops.get_dense_op_names([6212, 2001, 2002, 2003])
        self.add_node(3)
        self.add_node(4)
        self.add_edge(2, 3)
        self.add_edge(3, 4)
        self.v_edges.append((2,3))
        self.op_names.append([i for i in op_names_dense])
        self.v_edges.append((3,4))
        self.op_names.append([i for i in op_names_skip])
        dense_op_list = [(2,3)]
        skip_op_list = [(3,4)]
        self._set_cell_ops(dense_op_list, skip_op_list)
    
        '''
        #anchor_points = collections.deque([2], maxlen=3)
        for i in range(2, num_layers + 2):
            print('***********************************************', flush=True)
            print('i: ', i, flush=True)
            self.add_node(k)
            self.add_node(k+1)
            self.add_edge(k, k+1)
            print('point 1', flush=True)
            #TODO: add dense
            #self.edges[k, k+1].set("op", ops._get_dense_ops([2000, 2001, 2002, 2003]))
            dense_op_list.append((k, k+1))  
            self.v_edges.append((k, k+1))
            self.op_names.append(op_names_dense)
            self.add_node(k+2)
            self.add_edge(k+1, k+2)
            #self.edges[k+1, k+2].set("op", ops._get_identity_op())
            print('point 2', flush=True)
            k+=3
            old_k = k-1
            for anchor in anchor_points:
                print('inside anchor start it', flush=True)
                self.add_node(k)
                self.add_edge(k-1, k)
                #self.edges[k-1, k].set("op", [ops._get_zero_op(), ops._get_identity_op()]) 
                skip_op_list.append(( old_k, k))  
                self.v_edges.append((old_k, k))
                self.add_edge(old_k, k)
                self.op_names.append(op_names_skip)
                #self.edges[old_k, k].set("op", ops._get_identity_op()) 
                k+=1
                print('inside anchor done it', flush=True)
            print('point 3', flush=True)
            prev_input = old_k-1
            anchor_points.append(prev_input)
        self.add_edge(2,3)
        #self.edges[2,3].set("op", ops._get_identity_op())
        self._set_cell_ops(dense_op_list, skip_op_list)
        '''

    def get_all_op_names(self):
        return self.op_names
        
    def get_all_v_edges(self):
        return self.v_edges
    
    def _set_cell_ops(self, dense_op_list, skip_op_list) -> None:
        self.update_edges(
            update_func = lambda edge:self._set_dense_ops(edge),
            edges_to_update = dense_op_list,
            private_edge_data=True
        )
        self.update_edges(
            update_func = lambda edge:self._set_skip_ops(edge),
            edges_to_update = skip_op_list,
            private_edge_data=True
        )

    def _set_dense_ops(self, edge) -> None:
        edge.data.set(
            "op", ops._get_dense_ops([6212, 2001, 2002, 2003])
            )
    
    def _set_skip_ops(self, edge) -> None:
        edge.data.set(
            "op", [ops._get_zero_op(), ops._get_identity_op()]
            )
    
    def get_op_indices(self) -> list:
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def get_hash(self) -> tuple:
        return tuple(self.get_op_indices())

    #def get_arch_iterator(self, dataset_api=None) -> Iterator:
    #    return itertools.product(range(NUM_OPS), repeat=NUM_EDGES)

    def set_op_indices(self, op_indices: list) -> None:
        if self.instantiate_model == True:
            assert self.op_indices is None, f"An architecture has already been assigned to this instance of {self.__class__.__name__}. Instantiate a new instance to be able to sample a new model or set a new architecture."
        convert_op_indices_to_naslib(op_indices, self)
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
            print('SHOULD NOT be here (sampling labelled arch)!', flush=True)
            return self.sample_random_labeled_architecture()


        op_indices = []
        for i in range(len(self.v_edges)):
            rand_num = np.random.randint(len(self.op_names[i]))
            op_indices.append(rand_num)

        #op_indices = np.random.randint(NUM_OPS, size=(NUM_EDGES)).tolist()
        self.set_op_indices(op_indices)
        self.compact = self.get_op_indices()

    def mutate(self, parent: Graph, dataset_api: dict = None) -> None:
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = parent.get_op_indices()
        op_indices = list(parent_op_indices)

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in self.op_names[edge] if o != parent_op_indices[edge]]
        #available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
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
        available = [idx for idx, o in enumerate(self.op_names[edge]) if o != parent_op_indices[edge]]
        #available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
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
