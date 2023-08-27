import logging
import sys
import naslib as nl
import naslib
import copy
import numpy as np
import random
from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.defaults.trainer import Trainer
from naslib.optimizers.discrete.re.predictor_optimizer import RE_predictor
from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces.core.query_metrics import Metric
from naslib.runners._mpi import *
from naslib.utils.pytorch_helper import create_optimizer, create_criterion
from deephyper.evaluator import Evaluator
from naslib.utils.transfer_utils import *
from deephyper.nas.run._util import (
    load_config
)

import random
from datetime import datetime
from naslib.search_spaces import (
        NasBench101SearchSpace,
        NasBench201SearchSpace,
        NasBench301SearchSpace,
        NasBenchNLPSearchSpace,
        )
from naslib.search_spaces import get_search_space
from naslib.search_spaces.nasbench201.conversions import *
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api, get_zc_benchmark_api
from naslib import utils
import sys
sys.path.insert(1, '/home/ubuntu/experiments/')
sys.path.insert(1, '/home/ubuntu/experiments/cpp-store/')
sys.path.insert(1, '/home/ubuntu/')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False
torch.cuda.is_available = lambda : False
#sys.path.insert(1, '/home/mmadhya1/experiments/')
#sys.path.insert(1, '/home/mmadhya1/experiments/cpp-store/')
#sys.path.insert(1, '/home/mmadhya1/')
import experiments.transfer_learning as tl
import torch
import time
from mpi4py import MPI
from naslib.utils.pytorch_helper import create_optimizer, create_criterion
from torch.optim.lr_scheduler import CosineAnnealingLR
import uuid
import copy

torch.set_default_device('cuda')
if not MPI.Is_initialized():
    MPI.Init_thread()
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size


def query_zc_scores(arch, model_id=0, pred_graph=None, name_hash=None, transfer_method=None, timing_dict={}):
    #transfer_method = None
    zc_names = 'snip'
    zc = ZeroCost(method_type=zc_names)
    arch_hash = arch.get_hash()
    score = zc.query(arch,
            model_id=model_id,
            dataloader=train_queue, 
            pred_graph=pred_graph, 
            name_hash=name_hash, 
            transfer_method=transfer_method,
            timing_dict=timing_dict
            )
    return score

def train_epoch(config):
    loss = create_criterion(
            crit='CrossEntropyLoss',
    )
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arch = config['arch_seq']
    model = NasBench201SearchSpace()
    naslib.search_spaces.nasbench201.conversions.convert_op_indices_to_naslib(arch, model) 
    model.parse()
    op_optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0005,
    )
    scheduler = CosineAnnealingLR(op_optimizer,T_max = 32, eta_min = 0) 
    correct=0
    total=0
    final=0
    model_id = int(uuid.uuid4().int>>64)
    digraph = model.flatten_and_parse()
    name_hash = get_hashed_names(digraph, model)
    pred_graph = process_pred_graph(name_hash, digraph)
    if transfer_weights:
        transferred, parent_id = my_transfer_method.transfer(
            model, id=model_id, name_hash=name_hash, pred_graph=pred_graph, hint=None
        )
    counter=0
    for epoch in range(1):
        for step, data_train in enumerate(train_queue):
            data_test = next(iter(test_queue))
            device = torch.device('cuda') 
            data_train[0] = data_train[0].to(device)
            data_test[0] = data_test[0].to(device)
            data_train[1] = data_train[1].to(device)
            data_test[1] = data_test[1].to(device)
            model = model.to(device)
            op_optimizer.zero_grad()
            logits_train = model(data_train[0])
            train_loss = loss(logits_train, data_train[1])
            train_loss.backward()
            op_optimizer.step()
            scheduler.step()
            _, predicted = torch.max(logits_train, 1)
            total = data_train[1].size(0)
            correct = (predicted == data_train[1]).sum().item()
            final += float(correct) / float(total)
            counter += 1
    accuracy = float(final) / float(counter)
    if transfer_weights:
        my_transfer_method.store(
                              id=model_id,
                              model=model,
                              pred_graph=pred_graph,
                              name_hash=name_hash,
                              prefix=transferred,
                              val_acc=accuracy
                              )
    return accuracy


def query_nasbench201(arch_str, metric, dataset, dataset_api):
    if metric != Metric.RAW and metric != Metric.ALL:
        assert dataset in [
                "cifar10",
                "cifar100",
                "ImageNet16-120",
                ], "Unknown dataset: {}".format(dataset)
    if dataset_api is None:
        raise NotImplementedError("Must pass in dataset_api to query NAS-Bench-201")

    full_lc = False
    metric_to_nb201 = {
            Metric.TRAIN_ACCURACY: "train_acc1es",
            Metric.VAL_ACCURACY: "eval_acc1es",
            Metric.TEST_ACCURACY: "eval_acc1es",
            Metric.TRAIN_LOSS: "train_losses",
            Metric.VAL_LOSS: "eval_losses",
            Metric.TEST_LOSS: "eval_losses",
            Metric.TRAIN_TIME: "train_times",
            Metric.VAL_TIME: "eval_times",
            Metric.TEST_TIME: "eval_times",
            Metric.FLOPS: "flop",
            Metric.LATENCY: "latency",
            Metric.PARAMETERS: "params",
            Metric.EPOCH: "epochs",
            }

    if metric == Metric.RAW:
        # return all data
        return dataset_api["nb201_data"][arch_str]

    if dataset in ["cifar10", "cifar10-valid"]:
        query_results = dataset_api["nb201_data"][arch_str]
        # set correct cifar10 dataset
        dataset = "cifar10-valid"
    elif dataset == "cifar100":
        query_results = dataset_api["nb201_data"][arch_str]
    elif dataset == "ImageNet16-120":
        query_results = dataset_api["nb201_data"][arch_str]
    else:
        raise NotImplementedError("Invalid dataset")

    if metric == Metric.HP:
        # return hyperparameter info
        return query_results[dataset]["cost_info"]
    elif metric == Metric.TRAIN_TIME:
        return query_results[dataset]["cost_info"]["train_time"]

    if full_lc and epoch == -1:
        return query_results[dataset][metric_to_nb201[metric]]
    elif full_lc and epoch != -1:
        return query_results[dataset][metric_to_nb201[metric]][:epoch]
    else:
        # return the value of the metric only at the specified epoch
        return query_results[dataset][metric_to_nb201[metric]][epoch]


def model_size(model):
    param_size = 0
    for name, param in model.named_parameters():
        param_size += param.nelement() * param.element_size()
    #buffer_size = 0
    #for buffer in model.buffers():
    #    buffer_size += buffer.nelement() * buffer.element_size()

    #size_all_mb = (param_size) / 1024**2
    #size_all_mb = (param_size + buffer_size) / 1024**2
    return param_size
    #return size_all_mb


def query_nasbench301(
    self,
    metric: Metric = None,
    dataset: str = None,
    path: str = None,
    epoch: int = -1,
    full_lc: bool = False,
    dataset_api: dict = None):
    if dataset_api is None:
        raise NotImplementedError('Must pass in dataset_api to query NAS-Bench-301')

    assert dataset == 'cifar10' or dataset is None, "NAS-Bench-301 supports only CIFAR10 dataset"

    metric_to_nb301 = {
            Metric.TRAIN_LOSS: "train_losses",
            Metric.VAL_ACCURACY: "val_accuracies",
            Metric.TEST_ACCURACY: "val_accuracies",
            Metric.TRAIN_TIME: "runtime",
    }
    assert metric in [
        Metric.VAL_ACCURACY,
        Metric.TEST_ACCURACY,
        Metric.TRAIN_LOSS,
        Metric.TRAIN_TIME,
        Metric.HP,
    ], "Only VAL_ACCURACY, TEST_ACCURACY, TRAIN_LOSS, TRAIN_TIME, and HP can be queried for the given model."
    query_results = dataset_api["nb301_data"][self.compact]

    if metric == Metric.TRAIN_TIME:
        return query_results[metric_to_nb301[metric]]
    elif metric == Metric.HP:
                # todo: compute flops/params/latency for each arch. These are placeholders
        return {"flops": 15, "params": 0.1, "latency": 0.01}
    elif full_lc and epoch == -1:
        return query_results[metric_to_nb301[metric]]
    elif full_lc and epoch != -1:
        return query_results[metric_to_nb301[metric]][:epoch]
    else:
        # return the value of the metric only at the specified epoch
        return query_results[metric_to_nb301[metric]][epoch]


def str_to_tuple(arch):
    arch = arch.replace('[', '')
    arch = arch.replace(']', '')
    archs_list = arch.split(', ')
    archs_list = tuple([int(i) for i in archs_list])
    #arch_str = convert_op_indices_to_str(archs_list)
    return archs_list


def run_query_method(config):
    arch = config['arch']
    accuracy = query(arch, 
            Metric.TRAIN_ACCURACY,
            dataset, 
            dataset_api=dataset_api
            )
    return accuracy


def run_query_zc_method(config):
    arch = config['arch']
    if 'parent' in config:
        parent = config['parent']
        #dr = '/lus/grand/projects/VeloC/mmadhya1/pytorch_models/'
        #files = os.listdir(dr)
        #num_files = len(files)
        #filename = files[random.randint(0, num_files-1)]
        #parent = filename.split('.')[0]
    else:
        parent = None

    if search_space_str == 'nasbench201':
        if dataset == 'cifar10':
            model = NasBench201SearchSpace(n_classes=10)
        elif dataset == 'cifar100':
            model = NasBench201SearchSpace(n_classes=100)
        else: #imagenet
            model = NasBench201SearchSpace(n_classes=120)
        naslib.search_spaces.nasbench201.conversions.convert_op_indices_to_naslib(arch, model)
    elif search_space_str == 'nasbench301':
        model = NasBench301SearchSpace(n_classes=10)
        naslib.search_spaces.nasbench301.conversions.convert_compact_to_naslib(arch, model)
    elif search_space_str == 'candleattn':
        model = config['arch']
    else:
        raise Exception("Sorry, search space isn't supported") 
  
    Time = time.time()
    tint = int(str(Time)[-1:])
    

    rank = MPI.COMM_WORLD.rank
    random.seed(datetime.now().timestamp())
    #model_id = int(int(uuid.uuid4().int>>64) / 100) * rank
    tempid  = int(uuid.uuid4().int>>64) 
    tempid = int(tempid / 1000)
    tempid = int(tempid * 10)
    model_id = tempid + rank
    #model_id = int(uuid.uuid4().int>>64) 
    #print('model id: ', model_id, " rank: ", rank, flush=True)
    transfer_time = 0 
    #if parent is not None:
    if 1==2:
        print('parebt not none!', flush=True)
        with open(str(parent) + '.txt', "r") as file:
            op_index_str = file.read()

        op_tuple = str_to_tuple(op_index_str)
        #print('op_index_str: ', op_index_str, flush=True)
        model = NasBench201SearchSpace(n_classes=10)
        naslib.search_spaces.nasbench201.conversions.convert_op_indices_to_naslib(op_tuple, model)
        model.parse()
        digraph = model.flatten_and_parse()
        name_hash = get_hashed_names(digraph, model)
        time1 = time.time()
        transferred, parent_id = my_transfer_method.transfer(
            model, id=model_id, hint=parent
        )
        time2 = time.time()
        transfer_time = time2 - time1
        #print('transfer time: ', time2 - time1, flush=True)
    else:
        model.parse()
        digraph = model.flatten_and_parse()
        name_hash = get_hashed_names(digraph, model)
    
    pred_graph = process_pred_graph(name_hash, digraph)
    timing_dict1 = {}
    model_size_val = model_size(model)
    #print('model_size: ', model_size_val, flush=True)
    timing_dict1['model_size'] = model_size_val
    accuracy = query_zc_scores(
                            model,
                            model_id=model_id,
                            pred_graph=pred_graph,
                            name_hash=name_hash,
                            transfer_method=my_transfer_method,
                            timing_dict=timing_dict1
                )
    #lids = timing_dict1['cache_logs'][0]
    #sizes = timing_dict1['cache_logs'][1]
    #timings = timing_dict1['cache_logs'][2]
    #all_lids.append(lids)
    #all_sizes.append(sizes)
    #all_timings.append(timings)
    return accuracy, timing_dict1, model_id




global my_transfer_method

config = utils.get_config_from_args(config_type="nas_predictor")
transfer_method_cls = tl.transfer_methods.make_transfer_method(config.transfer_method, 'pytorch')


if config.transfer_method == 'datastates':
    setup = MPIExperimentSetup()
    setup.setup_gpus(ds_colocated=True)
    setup.workcomm, conn_strs = transfer_method_cls.startup_server(ds_colocated=True)
    my_transfer_method = transfer_method_cls(
        hosts=conn_strs,
        bulk_storage_path='/home/mmadhya1/experiments/',
        debug=False,
        store_weights=True
        )
else:
    my_transfer_method = transfer_method_cls(
        bulk_storage_path='/lus/grand/projects/VeloC/mmadhya1_2'
    )

#if not config.transfer_weights:
#    my_transfer_method=None
logger = setup_logger(config.save + "/log.log")
#logger.setLevel(logging.INFO)
utils.log_args(config)
utils.set_seed(config.seed)
eval_dict = {
        'train_epoch': train_epoch,
        'run_query_method': run_query_method,
        'run_query_zc_method': run_query_zc_method 
}

load_labeled = False
global dataset_api
global dataset
global train_queue
global valid_queue
global test_queue
global transfer_weights
global epoch
global search_space
global search_space_str


transfer_weights = config.transfer_weights
transfer_method = config.transfer_method
dataset_api = get_dataset_api(config.search_space, config.dataset)
search_space_str = config.search_space
dataset = config.dataset
epoch = config.train_epochs
train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
    config, mode="train"
)

eval_method_str = config.eval_method_str
eval_method = eval_dict[eval_method_str]
search_space = get_search_space(name=config.search_space, dataset=config.dataset)
trainer = Trainer(config, log_dir=config.log_dir, lightweight_output=True)
trainer.adapt_search_space(search_space, dataset_api=dataset_api)
print('done 1', flush=True)
'''
time.sleep(10)
if config.transfer_method == 'datastates':
    with Evaluator.create(
        eval_method,
        method='mpicomm',
        method_kwargs=setup.evaluator_method_kwargs(),
        ) as evaluator:
        if evaluator is not None:
            trainer.search(evaluator, resume_from="")
else:
    with Evaluator.create(
        eval_method,
        method='mpicomm',
        ) as evaluator:
        if evaluator is not None:
            trainer.search(evaluator, resume_from="")
'''

#print('all timings len: ', len(all_timings), flush=True)
#print('all sizes len: ', len(all_sizes), flush=True)
#print('all lids len: ', len(all_lids), flush=True)
#print(all_timings)
#print(arch_seq_baseline_dict, flush=True)
#import pickle

#with open('baseline_dict.pkl', 'wb') as f:
#    pickle.dump(arch_seq_baseline_dict, f)
#my_transfer_method.teardown_server(setup.workcomm)
