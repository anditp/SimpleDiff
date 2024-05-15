import torch 
from model import ScIDiff, Simple_Diff, ScIDiff_fourier, Simple_Diff_fourier
from MR_model import *
import os
from data.lagrangian_datatools import dataset_from_file
from learner import ScIDiffLearner
from torch.nn.parallel import DistributedDataParallel
import logger

def _train_impl(replica_id, model, dataset, params):
    torch.backends.cudnn.benchmark = True
    
    if params.type == "mr":
        opts = {}
        for i in range(params.levels):
            opts[i] = torch.optim.Adam(model[i].parameters(), lr=params.learning_rate)
        learner = MR_Full_Learner(params.model_dir, model, dataset, opts, params)
    
    elif params.type == "res_mr":
        opts = {}
        for i in range(params.levels):
            opts[i] = torch.optim.Adam(model[i].parameters(), lr=params.learning_rate)
        learner = MR_Res_Learner(params.model_dir, model, dataset, opts, params)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        
        if params.type == "sci_mr":
            learner = MR_Learner(params.model_dir, model, dataset, opt, params)
            
        else:
            learner = ScIDiffLearner(params.model_dir, model, dataset, opt, params)
    
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_steps=params.max_steps)

def train(model_params):
    dataset = dataset_from_file(model_params.data_path, model_params.batch_size, model_params.levels, coordinate=model_params.coordinate)
    model = ScIDiff(model_params).to(device='cuda' if torch.cuda.is_available() else 'cpu')
    _train_impl(0, model, dataset, model_params)


def train_distributed(replica_id, replica_count, port, model_params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
    dataset = dataset_from_file(model_params["data_path"], model_params["batch_size"], 
                                model_params["levels"], is_distributed=True,
                                fourier = False,
                                coordinate=model_params.coordinate)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    
    if model_params.type == "fourier":
        model = ScIDiff_fourier(model_params).to(device)
    elif model_params.type == "simple":
        model = Simple_Diff(model_params).to(device)
    elif model_params.type == "simple_fourier":
        model = Simple_Diff_fourier(model_params).to(device)
    elif model_params.type == "sci_mr":
        model = ScI_MR(model_params).to(device)
    elif model_params.type == "mr":
        models = {}
        for level in range(model_params.levels):
            models[level] = ScI_MR(model_params).to(device)
            models[level] = DistributedDataParallel(models[level], device_ids=[replica_id])
        _train_impl(replica_id, models, dataset, model_params)
        return
    elif model_params.type == "res_mr":
        models = {(model_params.levels - 1): ScI_MR_0(model_params).to(device)}
        for level in range(model_params.levels - 1):
            models[level] = ScI_MR_Res(model_params).to(device)
            models[level] = DistributedDataParallel(models[level], device_ids=[replica_id])
        _train_impl(replica_id, models, dataset, model_params)
        return
    else:
        model = ScIDiff(model_params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset, model_params)


    
    
    
    
    
    
    
    
    