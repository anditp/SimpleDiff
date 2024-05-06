import torch 
from model import ScIDiff
import os
from data.lagrangian_datatools import dataset_from_file
from learner import ScIDiffLearner
from torch.nn.parallel import DataParallel

    

def train(model_params):
    dataset = dataset_from_file(model_params["data_path"], model_params["batch_size"], model_params["levels"], is_distributed=True, num_workers=2, coordinate=model_params.coordinate)
    model = ScIDiff(model_params).to(device='cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    opt = torch.optim.Adam(model.parameters(), lr=model_params.learning_rate)
    learner = ScIDiffLearner(model_params.model_dir, model, dataset, opt, model_params)
    learner.train(max_steps=model_params.max_steps)

