"""
There are three representations
'naslib': the NASBench201SearchSpace object
'op_indices': A list of six ints, which is the simplest representation
'arch_str': The string representation used in the original nasbench201 paper

This file currently has the following conversions:
naslib -> op_indices
op_indices -> naslib
naslib -> arch_str

Note: we could add more conversions, but this is all we need for now
"""

import torch

from naslib.search_spaces.core.primitives import AbstractPrimitive

#OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
#OP_NAMES_NB201 = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']

#EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
#OPS_TO_NB201 = {
#    "AvgPool1x1": "avg_pool_3x3",
#    "ReLUConvBN1x1": "nor_conv_1x1",
#    "ReLUConvBN3x3": "nor_conv_3x3",
#    "Identity": "skip_connect",
#    "Zero": "none",
#}


def convert_naslib_to_op_indices(naslib_object):
    #child_graphs = naslib_object._get_child_graphs(single_instances=True)
    #print('child_graphs: ', child_graphs, flush=True)
    #cell = child_graphs[0]
    edge_list = naslib_object.get_all_v_edges()
    op_names = naslib_object.get_all_op_names()
    ops = []
    for i, j in edge_list:
        ops.append(naslib_object.edges[i, j]["op"].get_op_name())
        #ops.append(cell.edges[i, j]["op"].get_op_name())

    return [op_names[name_iter].index(name) for name_iter, name in enumerate(ops)]


def convert_op_indices_to_naslib(op_indices, naslib_object ):
    """
    Converts op indices to a naslib object
    input: op_indices (list of six ints)
    naslib_object is an empty NasBench201SearchSpace() object.
    Do not call this method with a naslib object that has already been
    discretized (i.e., all edges have a single op).

    output: none, but the naslib object now has all edges set
    as in genotype.

    warning: this method will modify the edges in naslib_object.
    """

    edge_list = naslib_object.get_all_v_edges()
    op_names = naslib_object.get_all_op_names()
    print('all edge list: ', edge_list, flush=True)
    print('all op names: ', op_names, flush=True)
    # create a dictionary of edges to ops
    edge_op_dict = {}
    for i, index in enumerate(op_indices):
        print('edge_list[i]: ', edge_list[i], flush=True)
        print('index: ', index, flush=True)
        edge_op_dict[edge_list[i]] = op_names[i][index]

    def add_op_index(edge):
        # function that adds the op index from the dictionary to each edge
        if (edge.head, edge.tail) in edge_op_dict:
            flag=0
            for i, op in enumerate(edge.data.op):
                print('op name: ', op.get_op_name(), flush=True)
                print('edge name: ', edge_op_dict[(edge.head, edge.tail)], flush=True)
                if op.get_op_name() == edge_op_dict[(edge.head, edge.tail)]:
                    flag = 1
                    index = i
                    break
            if flag == 1:
                edge.data.set("op_index", index, shared=True)

    def update_ops(edge):
        # function that replaces the primitive ops at the edges with the one in op_index
        print('edge.data: ', edge.data, flush=True)
        print('edge.head: ', edge.head, flush=True)
        print('edge.tail: ', edge.tail, flush=True)
        if isinstance(edge.data.op, list):
            primitives = edge.data.op
        else:
            primitives = edge.data.primitives

        print('primitives: ', primitives, flush=True)
        chosen_op = primitives[edge.data.op_index]
        primitives[edge.data.op_index] = update_batchnorms(chosen_op)

        edge.data.set("op", primitives[edge.data.op_index])
        edge.data.set("primitives", primitives)  # store for later use

    def update_batchnorms(op: AbstractPrimitive) -> AbstractPrimitive:
        """ Makes batchnorms in the op affine, if they exist """
        init_params = op.init_params
        has_batchnorm = False

        for module in op.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                has_batchnorm = True
                break

        if not has_batchnorm:
            return op

        if 'affine' in init_params:
            init_params['affine'] = True
        if 'track_running_stats' in init_params:
            init_params['track_running_stats'] = True

        new_op = type(op)(**init_params)
        return new_op

    naslib_object.update_edges(
        add_op_index, edges_to_update=edge_list ,private_edge_data=False
    )

    naslib_object.update_edges(
        update_ops, edges_to_update=edge_list,private_edge_data=True
    )
