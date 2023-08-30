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
    print('emter snip function', flush=True)
    #inputs = torch.transpose(inputs, 0,1)
    total_start = time.time()
    targets = torch.argmax(targets, dim=1)
    print('target shape: ', targets.shape, flush=True) 
    print(targets, flush=True)
    #model_id = int(uuid.uuid4().int>>64)
    #print('model id: ', model_id, flush=True)
    
    #start_transfer = time.time()
    #transferred, parent_id = transfer_method.transfer(
    #    net, id=model_id, hint=name_hash['parent']
    #)
    #end_transfer = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('input shape: ', inputs.shape[0], flush=True)
    net(torch.rand(inputs.shape))
    print('emter snip function 1', flush=True)
    #print(device, flush=True)
    inputs, targets = inputs.to(device), targets.to(device)
    print('emter snip function 2', flush=True)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            #layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    print('emter snip function 3', flush=True)
    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    transferred = []
    start_transfer = time.time()
    all_lids = []
    all_sizes = []
    all_timings = []
    if transfer_method:
        #transferred, parent_id, lids, sizes, timings = transfer_method.transfer(
        #transferred, parent_id = transfer_method.transfer(
        #    net, id=model_id, name_hash=name_hash, pred_graph=pred_graph, hint=None
        #)
        timing_dict['transferred'] = 0 
        
        #if len(lids) > 0:
        #    all_lids.extend(lids)
        #    all_sizes.extend(sizes)
        #    all_timings.extend(timings)

    end_transfer = time.time()
    print('emter snip function 4', flush=True)
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        print('emter snip function 4.1', flush=True)
        outputs = net.forward(inputs[st:en])
        print('emter snip function 4.2', flush=True)
        loss = loss_fn(outputs, targets[st:en])
        print('emter snip function 4.3', flush=True)
        loss.backward()
        print('emter snip function 4.4', flush=True)

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)

    print('emter snip function 5.0', flush=True)
    grads_abs = get_layer_metric_array(net, snip, mode)
    print('emter snip function 5', flush=True)

    #print('number transferred: ', len(transferred), flush=True)

    if transfer_method:
        sum1 = 0.0
        #for i in range(len(grads_abs)):
        #    sum1 += torch.sum(grads_abs[i])
        #sum1  = sum1.item()
        start_store = time.time()
        #lids_store, sizes_store, timings_store = transfer_method.store(
        #transfer_method.store(
        #                      id=model_id, 
        #                      model=net, 
        #                      name_hash=name_hash, 
        #                      pred_graph=pred_graph, 
        #                      prefix=transferred, 
        #                      val_acc=sum1
        #                      )
        
        #all_lids.extend(lids_store)
        #all_sizes.extend(sizes_store)
        #all_timings.extend(timings_store)
        end_store = time.time()
        #timing_dict['stored'] = slu 
        timing_dict['store_time'] = end_store - start_store

    timing_dict['transfer_time'] = end_transfer - start_transfer
    timing_dict['train_time'] = end_store - total_start
    print('store time: ', timing_dict['store_time'], flush=True)
    print('transfer time: ', timing_dict['transfer_time'], flush=True)
    #timing_dict['cache_logs'] = [all_timings, all_lids, all_sizes] 
    #print("store time: ", timing_dict['store_time'], flush=True)
    #print("transfer time: ", timing_dict['transfer_time'], flush=True)

    return grads_abs
