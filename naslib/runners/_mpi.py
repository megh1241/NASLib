import logging
import os
import abc
import torch

class MPIExperimentSetup(metaclass=abc.ABCMeta):
    def __init__(self, *_args, **_kwargs):
        super().__init__()

    def setup_gpus(self, ds_colocated=False):
        import mpi4py

        mpi4py.rc.initialize = False
        mpi4py.rc.threads = True
        mpi4py.rc.thread_level = "multiple"
        from mpi4py import MPI

        if not MPI.Is_initialized():
            MPI.Init_thread()

        '''
        gpu_per_node = 4
        if ds_colocated:
            rank = MPI.COMM_WORLD.Get_rank()
            num_nodes = MPI.COMM_WORLD.Get_size()  / 6
            gpu_local_idx = (int(rank/num_nodes)-1) % gpu_per_node
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            gpu_local_idx = (rank-1) % gpu_per_node
        '''
        gpu_per_node = 1
        rank = MPI.COMM_WORLD.Get_rank()
        gpu_local_idx = (rank) % gpu_per_node
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_local_idx) 
    
    def evaluator_method_kwargs(self):
        return {"comm": self.workcomm}

    def trainer_method_kwargs(self):
        return {}
