import logging
import sys
import naslib as nl

from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.defaults.trainer import Trainer
from naslib.optimizers.discrete.re.predictor_optimizer import RE_predictor
from naslib.search_spaces.core.query_metrics import Metric
#from naslib.predictors import OneShotPredictor
from naslib.runners._mpi import *

from deephyper.evaluator import Evaluator

from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    NasBenchNLPSearchSpace,
)
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api, get_zc_benchmark_api
import sys
sys.path.insert(1, '/home/mmadhya1/experiments/')
sys.path.insert(1, '/home/mmadhya1/experiments/cpp-store/')
sys.path.insert(1, '/home/mmadhya1/')
import experiments.transfer_learning as tl

from mpi4py import MPI
if not MPI.Is_initialized():
    MPI.Init_thread()
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

setup = MPIExperimentSetup()
setup.setup_gpus(ds_colocated=True)


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


supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "nasbench301": NasBench301SearchSpace(),
    "nlp": NasBenchNLPSearchSpace(),
}


load_labeled = False
global dataset_api
global dataset

dataset_api = get_dataset_api(config.search_space, config.dataset)
dataset = config.dataset
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]
trainer = Trainer(config, log_dir=config.log_dir, lightweight_output=True)
trainer.adapt_search_space(search_space, dataset_api=dataset_api)
with Evaluator.create(
    run_query_method,
    method='mpicomm',
    method_kwargs=setup.evaluator_method_kwargs(),
) as evaluator:
    if evaluator is not None:
        trainer.search(evaluator, resume_from="")
my_transfer_method.teardown_server(setup.workcomm)
