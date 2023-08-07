# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types
import uuid
import time

from . import measure
from ..p_utils import get_layer_metric_array


def snip_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


@measure("snip", bn=True, mode="param")
def compute_snip_per_weight(net, inputs, targets, mode, loss_fn, model_id=None,split_data=1, pred_graph=None, name_hash=None, transfer_method=None, timing_dict={}):
    total_start = time.time()
    #model_id = int(uuid.uuid4().int>>64)
    #print('model id: ', model_id, flush=True)
    
    #start_transfer = time.time()
    #transferred, parent_id = transfer_method.transfer(
    #    net, id=model_id, hint=name_hash['parent']
    #)
    #end_transfer = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs, targets = inputs.to(device), targets.to(device)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    start_transfer = time.time()
    if name_hash['parent'] != None:
        del name_hash['parent']
        #transferred, _ = transfer_method.transfer(
        #    net, id=model_id, name_hash=name_hash, pred_graph=pred_graph, hint=None
        #)
    else:
        del name_hash['parent']
        transferred = []
    if transfer_method:
        #transferred, parent_id = transfer_method.transfer(
        #    net, id=model_id, hint=name_hash['parent']
        #)
        transferred, parent_id, trl = transfer_method.transfer(
            net, id=model_id, name_hash=name_hash, pred_graph=pred_graph, hint=None
        )
        timing_dict['transferred'] = len(trl) 
    end_transfer = time.time()
    #end_transfer = time.time()
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, snip, mode)
    
    if transfer_method:
        sum1 = 0.0
        for i in range(len(grads_abs)):
            sum1 += torch.sum(grads_abs[i])
        sum1  = sum1.item()
        start_store = time.time()
        ''' 
        transfer_method.store(
                              id=model_id,
                              model=net,
                              )
        ''' 
        slu = transfer_method.store(
                              id=model_id, 
                              model=net, 
                              name_hash=name_hash, 
                              pred_graph=pred_graph, 
                              prefix=transferred, 
                              val_acc=sum1
                              )
        end_store = time.time()
        timing_dict['stored'] = slu 
        timing_dict['store_time'] = end_store - start_store

    timing_dict['transfer_time'] = end_transfer - start_transfer
    timing_dict['train_time'] = end_store - total_start
    print("store time: ", timing_dict['store_time'], flush=True)
    print("transfer time: ", timing_dict['transfer_time'], flush=True)

    return grads_abs
