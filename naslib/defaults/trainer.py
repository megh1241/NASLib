import codecs
import collections
from naslib.search_spaces.core.graph import Graph
from naslib.utils.vis import plot_architectural_weights
import time
import json
import logging
import os
import copy
import torch
import numpy as np
import naslib
import csv
from fvcore.common.checkpoint import PeriodicCheckpointer

from naslib.search_spaces.core.query_metrics import Metric

from naslib import utils
from naslib.utils.log import log_every_n_seconds, log_first_n

from typing import Callable
from .additional_primitives import DropPathWrapper

logger = logging.getLogger(__name__)
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpaceMicro,
    TransBench101SearchSpaceMacro,
    NasBenchASRSearchSpace
)
#from ._simplehdf5_pytorch import TransferSimpleHDF5Pytorch

class Trainer(object):
    """
    Default implementation that handles dataloading and preparing batches, the
    train loop, gathering statistics, checkpointing and doing the final
    final evaluation.

    If this does not fulfil your needs free do subclass it and implement your
    required logic.
    """

    def __init__(self, config, log_dir='.', optimizer=None, lightweight_output=False):
        """
        Initializes the trainer.

        Args:
            optimizer: A NASLib optimizer
            config (AttrDict): The configuration loaded from a yaml file, e.g
                via  `utils.get_config_from_args()`
        """
        self._log_dir = log_dir
        self.optimizer = optimizer
        self.config = config
        self.epochs = self.config.search.epochs
        self.lightweight_output = lightweight_output
        self._sample_size = config.search.sample_size
        self._population_size = config.search.population_size
        self._population = collections.deque(maxlen=self._population_size)
        self._random_state = np.random.RandomState()
        # preparations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # measuring stuff
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_loss = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_loss = utils.AverageMeter()
        self.cache_dict = [[], [], []]
        n_parameters = 0
        #n_parameters = optimizer.get_model_size()
        # logger.info("param size = %fMB", n_parameters)
        self.search_trajectory = utils.AttrDict(
            {
                "train_acc": [],
                "train_loss": [],
                "valid_acc": [],
                "valid_loss": [],
                "test_acc": [],
                "test_loss": [],
                "runtime": [],
                "train_time": [],
                "arch_eval": [],
                "params": n_parameters,
            }
        )

    def adapt_search_space(self, search_space: Graph, scope: str = None, dataset_api: dict = None):
        assert (
            search_space.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api


    def _gen_random_batch(self, size):
        batch = []
        for _ in range(size):
            cfg = {}
            model = (
                torch.nn.Module()
            )
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            #TODO change this maybe
            if self.config.search_space == 'nasbench201':
                cfg['arch'] = naslib.search_spaces.nasbench201.conversions.convert_naslib_to_op_indices(model.arch)
                cfg['arch_seq'] = naslib.search_spaces.nasbench201.conversions.convert_naslib_to_op_indices(model.arch)
            elif self.config.search_space == 'nasbench301':
                cfg['arch'] = naslib.search_spaces.nasbench301.conversions.convert_naslib_to_compact(model.arch)
            elif self.config.search_space == 'candleattn':
                val = naslib.search_spaces.candleattn.conversions.convert_naslib_to_op_indices(model.arch)
                cfg['arch'] = val
                cfg['arch_seq'] = val
            batch.append(cfg)
        return batch


    def _saved_keys(self, job):
        res = {"arch_seq": job.config["arch_seq"] }
        if 'timing_dict' in job.config:
            keys = ['train_time', 'transfer_time', 'store_time', 'transferred', 'stored']
            for key in keys:
                res[key] = job.config["timing_dict"][key]
        return res


    def search(self, evaluator=None, resume_from="", summary_writer=None, after_epoch: Callable[[int], None]=None, report_incumbent=True):
        """
        Start the architecture search.

        Generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                train from scratch.
        """
        logger.info("Beginning search")
        self._evaluator = evaluator
        np.random.seed(self.config.search.seed)
        torch.manual_seed(self.config.search.seed)
        
        num_evals_done = 0
        batch = self._gen_random_batch(size=self._evaluator.num_workers)
        self._evaluator.submit(batch) 
     
        while self.epochs < 0  or num_evals_done < self.epochs:
            new_results = self._evaluator.gather("BATCH", 1)
            num_received = len(new_results)
            if num_received > 0:
                self._population.extend(new_results)
                #cache_dict_curr = new_results[0].result[1]['cache_logs']
                #self.cache_dict[0].extend(cache_dict_curr[0])
                #self.cache_dict[1].extend(cache_dict_curr[1])
                #self.cache_dict[2].extend(cache_dict_curr[2])
                #self._evaluator.dump_evals(
                #    saved_keys=self._saved_keys, timings_dict=new_results[0].result[1], log_dir=self._log_dir
                #)
                num_evals_done += num_received
                if num_evals_done >= self.epochs:
                    break
                # If the population is big enough evolve the population
                if len(self._population) == self._population_size:
                    children_batch = []
                    # For each new parent/result we create a child from it
                    for _ in range(num_received):
                        # select_sample
                        indexes = self._random_state.choice(
                            self._population_size, self._sample_size, replace=False
                        )

                        sample = [self._population[i] for i in indexes]
                        # select_parent
                        cfg, vals = max(sample,  key=lambda x: x[1][0])
                        # copy_mutate_parent
                        child = (
                            torch.nn.Module()
                        )
                        child_cfg = {}
                        child_cfg['arch_seq'] = 'arch'
                        if self.config.search_space == 'nasbench201':
                            child.arch = self.search_space.clone()
                            child.arch.mutate_with_parent_op_indices(cfg['arch'], dataset_api=self.dataset_api)
                            child_cfg['arch'] = naslib.search_spaces.nasbench201.conversions.convert_naslib_to_op_indices(child.arch)
                            child_cfg['arch_seq'] = naslib.search_spaces.nasbench201.conversions.convert_naslib_to_op_indices(child.arch)
                        elif self.config.search_space == 'nasbench301':
                            parent = NasBench301SearchSpace(n_classes=10)
                            child.arch = parent
                            child.arch.mutate(cfg['arch'], dataset_api=self.dataset_api)
                            child_cfg['arch'] = naslib.search_spaces.nasbench301.conversions.convert_naslib_to_compact(child.arch)
                            child_cfg['arch_seq'] = 'arch'
                        elif self.config.search_space == 'candleattn':
                            child.arch = self.search_space.clone()
                            child.arch.mutate_with_parent_op_indices(cfg['arch'], dataset_api=self.dataset_api)
                            val = naslib.search_spaces.candleattn.conversions.convert_naslib_to_op_indices(child.arch)
                            child_cfg['arch'] = val
                            child_cfg['arch_seq'] = val
                        else:
                            raise Exception('Sorry the search space is not supported')
                        
                        child_cfg['parent'] = vals[2]
                        children_batch.append(child_cfg)
                    
                    # submit_childs
                    self._evaluator.submit(children_batch)

                else:  # If the population is too small keep increasing it
                    self._evaluator.submit(self._gen_random_batch(size=num_received))

        '''
        print('printing cache dict: ', self.cache_dict, flush=True)
        if len(self.cache_dict[0]) > 0:
            min_ele = min(self.cache_dict[0])
            self.cache_dict[0] = [ele - min_ele for ele in self.cache_dict[0]]
            min_ele = min(self.cache_dict[1])
            self.cache_dict[1] = [ele - min_ele for ele in self.cache_dict[1]]
            timings_to_store = zip(*self.cache_dict)
            
            with open('timing_results.csv','w') as out:
                csv_out=csv.writer(out, delimiter=' ')
                for row in timings_to_store:
                    csv_out.writerow(row)
        '''
        logger.info("Training finished")

    def evaluate_oneshot(self, resume_from="", dataloader=None):
        """
        Evaluate the one-shot model on the specified dataset.

        Generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                evaluate with the current one-shot weights.
        """
        logger.info("Start one-shot evaluation")
        self.optimizer.before_training()
        self._setup_checkpointers(resume_from)

        loss = torch.nn.CrossEntropyLoss()

        if dataloader is None:
            # load only the validation data
            _, dataloader, _ = self.build_search_dataloaders(self.config)

        self.optimizer.graph.eval()
        with torch.no_grad():
            start_time = time.time()
            for step, data_val in enumerate(dataloader):
                input_val = data_val[0].to(self.device)
                target_val = data_val[1].to(self.device, non_blocking=True)

                logits_val = self.optimizer.graph(input_val)
                val_loss = loss(logits_val, target_val)

                self._store_accuracies(logits_val, data_val[1], "val")
                self.val_loss.update(float(val_loss.detach().cpu()))

            end_time = time.time()

            self.search_trajectory.valid_acc.append(self.val_top1.avg)
            self.search_trajectory.valid_loss.append(self.val_loss.avg)
            self.search_trajectory.runtime.append(end_time - start_time)

            self._log_to_json()

        logger.info("Evaluation finished")
        return self.val_top1.avg

    def evaluate(
        self,
        retrain:bool=True,
        search_model:str="",
        resume_from:str="",
        best_arch:Graph=None,
        dataset_api:object=None,
        metric:Metric=None,
    ):
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        Args:
            retrain (bool)      : Reset the weights from the architecure search
            search_model (str)  : Path to checkpoint file that was created during search. If not provided,
                                  then try to load 'model_final.pth' from search
            resume_from (str)   : Resume retraining from the given checkpoint file.
            best_arch           : Parsed model you want to directly evaluate and ignore the final model
                                  from the optimizer.
            dataset_api         : Dataset API to use for querying model performance.
            metric              : Metric to query the benchmark for.
        """
        #print("Start evaluation")
        #logger.info("Start evaluation")
        if not best_arch:
            if not search_model:
                search_model = os.path.join(
                    self.config.save, "search", "model_final.pth"
                )
            #self._setup_checkpointers(search_model)  # required to load the architecture

            best_arch = self.optimizer.get_final_architecture()
        logger.info(f"Final architecture hash: {best_arch.get_hash()}")

        if metric is None:
            metric = Metric.TEST_ACCURACY
        
        result = best_arch.query(
            metric=metric, dataset=self.config.dataset, dataset_api=dataset_api
        )
        
        logger.info("Queried results ({}): {}".format(metric, result))
        return result

    @staticmethod
    def build_search_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="train"
        )
        return train_queue, valid_queue, _  # test_queue is not used in search currently

    @staticmethod
    def build_eval_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="val"
        )
        return train_queue, valid_queue, test_queue

    @staticmethod
    def build_eval_optimizer(parameters, config):
        return torch.optim.SGD(
            parameters,
            lr=config.evaluation.learning_rate,
            momentum=config.evaluation.momentum,
            weight_decay=config.evaluation.weight_decay,
        )

    @staticmethod
    def build_search_scheduler(optimizer, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.search.epochs,
            eta_min=config.search.learning_rate_min,
        )

    @staticmethod
    def build_eval_scheduler(optimizer, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.evaluation.epochs,
            eta_min=config.evaluation.learning_rate_min,
        )

    def _log_and_reset_accuracies(self, epoch, writer=None):
        logger.info(
            "Epoch {} done. Train accuracy: {:.5f}, Validation accuracy: {:.5f}".format(
                epoch,
                self.train_top1.avg,
                self.val_top1.avg,
            )
        )

        if writer is not None:
            writer.add_scalar('Train accuracy (top 1)', self.train_top1.avg, epoch)
            writer.add_scalar('Train accuracy (top 5)', self.train_top5.avg, epoch)
            writer.add_scalar('Train loss', self.train_loss.avg, epoch)
            writer.add_scalar('Validation accuracy (top 1)', self.val_top1.avg, epoch)
            writer.add_scalar('Validation accuracy (top 5)', self.val_top5.avg, epoch)
            writer.add_scalar('Validation loss', self.val_loss.avg, epoch)

        self.train_top1.reset()
        self.train_top5.reset()
        self.train_loss.reset()
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_loss.reset()

    def _store_accuracies(self, logits, target, split):
        """Update the accuracy counters"""
        logits = logits.clone().detach().cpu()
        target = target.clone().detach().cpu()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = logits.size(0)

        if split == "train":
            self.train_top1.update(prec1.data.item(), n)
            self.train_top5.update(prec5.data.item(), n)
        elif split == "val":
            self.val_top1.update(prec1.data.item(), n)
            self.val_top5.update(prec5.data.item(), n)
        else:
            raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")

    def _prepare_dataloaders(self, config, mode="train"):
        """
        Prepare train, validation, and test dataloaders with the splits defined
        in the config.

        Args:
            config (AttrDict): config from config file.
        """
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode
        )
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue

    def _setup_checkpointers(
        self, resume_from="", search=True, period=1, **add_checkpointables
    ):
        """
        Sets up a periodic chechkpointer which can be used to save checkpoints
        at every epoch. It will call optimizer's `get_checkpointables()` as objects
        to store.

        Args:
            resume_from (str): A checkpoint file to resume the search or evaluation from.
            search (bool): Whether search or evaluation phase is checkpointed. This is required
                because the files are in different folders to not be overridden
            add_checkpointables (object): Additional things to checkpoint together with the
                optimizer's checkpointables.
        """
        checkpointables = self.optimizer.get_checkpointables()
        checkpointables.update(add_checkpointables)

        checkpointer = utils.Checkpointer(
            model=checkpointables.pop("model"),
            save_dir=self.config.save + "/search"
            if search
            else self.config.save + "/eval",
            # **checkpointables #NOTE: this is throwing an Error
        )

        self.periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=period,
            max_iter=self.config.search.epochs
            if search
            else self.config.evaluation.epochs,
        )

        if resume_from:
            logger.info("loading model from file {}".format(resume_from))
            checkpoint = checkpointer.resume_or_load(resume_from, resume=True)
            if checkpointer.has_checkpoint():
                return checkpoint.get("iteration", -1) + 1
        return 0

    def _log_to_json(self):
        """log training statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        if not self.lightweight_output:
            with codecs.open(
                os.path.join(self.config.save, "errors.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(self.search_trajectory, file, separators=(",", ":"))
        else:
            with codecs.open(
                os.path.join(self.config.save, "errors.json"), "w", encoding="utf-8"
            ) as file:
                lightweight_dict = copy.deepcopy(self.search_trajectory)
                for key in ["arch_eval", "train_loss", "valid_loss", "test_loss"]:
                    lightweight_dict.pop(key)
                json.dump([self.config, lightweight_dict], file, separators=(",", ":"))
