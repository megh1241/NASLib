import collections
import logging
import torch
import copy
import numpy as np


from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.predictors.zerocost import ZeroCost

from naslib.utils import count_parameters_in_MB, get_train_val_loaders
from naslib.utils.transfer_utils import *
from naslib.utils.log import log_every_n_seconds

from fvcore.common.config import CfgNode
from naslib.optimizers.discrete.re.optimizer import RegularizedEvolution


logger = logging.getLogger(__name__)

class RE_predictor(RegularizedEvolution):
    def __init__(self, config, transfer_method=None, zc_api=None):
        super().__init__(config)
        self.transfer_method = transfer_method
        self.zc_api = zc_api
        self.use_zc_api = config.search.use_zc_api if hasattr(
            config.search, 'use_zc_api') else False
        self.zc_names = config.search.zc_names if hasattr(
            config.search, 'zc_names') else None
        self.zc_only = config.search.zc_only if hasattr(
            config.search, 'zc_only') else False


    def adapt_search_space(self, search_space: Graph, scope: str = None, dataset_api: dict = None):
        assert (
            search_space.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api 
        self.train_loader, _, _, _, _ = get_train_val_loaders(
            self.config, mode="train")


    def get_zero_cost_predictors(self):
        return {self.zc_names: ZeroCost(method_type=self.zc_names)}
        #return {zc_name: ZeroCost(method_type=zc_name) for zc_name in self.zc_names}


    def query_zc_scores(self, arch, pred_graph=None, name_hash=None, transfer_method=None):
        zc_scores = {}
        zc_methods = self.get_zero_cost_predictors()
        arch_hash = arch.get_hash()
        for zc_name, zc_method in zc_methods.items():

            if self.use_zc_api and str(arch_hash) in self.zc_api:
                score = self.zc_api[str(arch_hash)][zc_name]['score']
            else:
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                score = zc_method.query(arch, dataloader=zc_method.train_loader, pred_graph=pred_graph, name_hash=name_hash, transfer_method=transfer_method)

            if float("-inf") == score:
                score = -1e9
            elif float("inf") == score:
                score = 1e9

            zc_scores[zc_name] = score

        return zc_scores


    def new_epoch(self, epoch: int):
        # We sample as many architectures as we need
        if epoch < self.population_size:
            logger.info("Start sampling architectures to fill the population")
            # If there is no scope defined, let's use the search space default one

            model = (
                torch.nn.Module()
            )
            
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.arch.parse()
            digraph = model.arch.flatten_and_parse()
            
            if self.config.transfer_weights:
                hashed_names = get_hashed_names(digraph, model.arch)
                pred_graph_processed = process_pred_graph(hashed_names, digraph)
                model.accuracy = self.query_zc_scores(model.arch, 
                                                pred_graph=pred_graph_processed, 
                                                name_hash=hashed_names, 
                                                transfer_method=self.transfer_method)
            else:
                model.accuracy = self.query_zc_scores(model.arch)
            self.population.append(model)
            self._update_history(model)
            log_every_n_seconds(
                logging.INFO, "Population size {}".format(len(self.population))
            )
        else:
            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)

            parent = max(sample, key=lambda x: x.accuracy[self.zc_names])

            child = (
                torch.nn.Module()
            )
            child.arch = self.search_space.clone()
            child.arch.mutate(parent.arch, dataset_api=self.dataset_api)
            child.arch.parse()
            digraph = child.arch.flatten_and_parse()
            if self.config.transfer_weights:
                hashed_names = get_hashed_names(digraph, child.arch)
                pred_graph_processed = process_pred_graph(hashed_names, digraph)
                child.accuracy = self.query_zc_scores(child.arch, 
                                                pred_graph=pred_graph_processed, 
                                                name_hash=hashed_names, 
                                                transfer_method=self.transfer_method)
            else:
                child.accuracy = self.query_zc_scores(child.arch)
            self.population.append(child)
            self._update_history(child)

    def get_final_architecture(self):
        #print('printing history: ', self.history, flush=True)
        return max(self.history, key=lambda x: x.accuracy[self.zc_names]).arch
