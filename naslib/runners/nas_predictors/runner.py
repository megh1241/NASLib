import logging
import sys
import naslib as nl
import naslib
from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.defaults.trainer import Trainer
from naslib.optimizers.discrete.re.predictor_optimizer import RE_predictor
from naslib.search_spaces.core.query_metrics import Metric
from naslib.runners._mpi import *
from naslib.utils.pytorch_helper import create_optimizer, create_criterion
from deephyper.evaluator import Evaluator
from naslib.utils.transfer_utils import *
from naslib.search_spaces import (
        NasBench101SearchSpace,
        NasBench201SearchSpace,
        NasBench301SearchSpace,
        NasBenchNLPSearchSpace,
        )
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api, get_zc_benchmark_api
from naslib import utils
import sys
sys.path.insert(1, '/home/mmadhya1/experiments/')
sys.path.insert(1, '/home/mmadhya1/experiments/cpp-store/')
sys.path.insert(1, '/home/mmadhya1/')
import experiments.transfer_learning as tl
import torch
from mpi4py import MPI
from naslib.utils.pytorch_helper import create_optimizer, create_criterion
from torch.optim.lr_scheduler import CosineAnnealingLR
import uuid

if not MPI.Is_initialized():
    MPI.Init_thread()
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

setup = MPIExperimentSetup()
setup.setup_gpus(ds_colocated=True)


def train_epoch(config):
    loss = create_criterion(
            crit='CrossEntropyLoss',
    )

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
            data_train = (
                data_train[0].to(device),
                data_train[1].to(device, non_blocking=True),
            )
            data_test = next(iter(test_queue))
            data_val = (
                        data_val[0].to(device),
                        data_val[1].to(device, non_blocking=True),
            )
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


def query(arch_str, metric, dataset, dataset_api):
    epoch=1
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


def run_query_method(config):
    arch = config['arch']
    accuracy = query(arch, 
            Metric.TRAIN_ACCURACY,
            dataset, 
            dataset_api=dataset_api
            )
    return accuracy



global my_transfer_method
transfer_method_cls = tl.transfer_methods.make_transfer_method('datastates', 'pytorch')
setup.workcomm, conn_strs = transfer_method_cls.startup_server(ds_colocated=True)
my_transfer_method = transfer_method_cls(
        hosts=conn_strs,
        bulk_storage_path='/home/mmadhya1/experiments/',
        debug=False,
        store_weights=True
        )
config = utils.get_config_from_args(config_type="nas_predictor")
if not config.transfer_weights:
    my_transfer_method=None

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)
utils.set_seed(config.seed)

supported_search_spaces = {
        "nasbench101": NasBench101SearchSpace(),
        "nasbench201": NasBench201SearchSpace(),
        "nasbench301": NasBench301SearchSpace(),
        "nlp": NasBenchNLPSearchSpace(),
        }


load_labeled = False
global dataset_api
global dataset
global train_queue
global valid_queue
global test_queue
global transfer_weights
global device

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
device = available_gpus[0]
transfer_weights = config.transfer_weights
dataset_api = get_dataset_api(config.search_space, config.dataset)
dataset = config.dataset
train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
    config, mode="train"
)

search_space = supported_search_spaces[config.search_space]
trainer = Trainer(config, log_dir=config.log_dir, lightweight_output=True)
trainer.adapt_search_space(search_space, dataset_api=dataset_api)
with Evaluator.create(
        #train_epoch,
        run_query_method,
        method='mpicomm',
        method_kwargs=setup.evaluator_method_kwargs(),
        ) as evaluator:
    if evaluator is not None:
        trainer.search(evaluator, resume_from="")
my_transfer_method.teardown_server(setup.workcomm)
