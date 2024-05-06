from argparse import ArgumentParser
from torch.cuda import device_count
import torch
from torch.multiprocessing import spawn

# Monkey patch collections
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from training import train, train_distributed
import yaml
from attrdict import AttrDict
import os
import logger

def _get_free_port():
  import socketserver
  with socketserver.TCPServer(("localhost", 0), None) as s:
    return s.server_address[1]


def main(args):
    logger.configure(dir = "logs")

    replica_count = device_count()
    # obtain configuration file
    with open(args.params_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)  # config is dict
    cfg = AttrDict(config)
    model_params = cfg.parameters
    model_params.data_path = args.dataset_path
    model_params.model_dir = args.experiment_dir
  
    assert model_params.coordinate >= -1 and model_params.coordinate <=2 and type(model_params.coordinate) == int
    
    model_params.coordinate = None if model_params.coordinate==-1 else model_params.coordinate
    model_params.num_coords = 3 if model_params.coordinate is None else 1
    logger.log(model_params)
    # dump config file to experiment directory
    with open(os.path.join(args.experiment_dir,"params.yaml"), "w") as f:
        yaml.dump(config, f)
  
    logger.log(torch.cuda.is_available())
    if replica_count > 1:
        if model_params.batch_size % replica_count != 0:
            raise ValueError(f"Batch size {model_params.batch_size} is not evenly divisble by # GPUs {replica_count}.")
        model_params.batch_size = model_params.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, model_params), nprocs=replica_count, join=True)
    else:
        train(model_params)


if __name__ == "__main__":
  parser = ArgumentParser(description="train (or resume training) a model")
  parser.add_argument("experiment_dir",
      help="directory in which to store model checkpoints and training logs")
  parser.add_argument("dataset_path",
                      help="path to the .npy file containing the dataset")
  parser.add_argument("-p", "--params_path", default="./params.yaml",
      help="path to the file containing model parameters")
  main(parser.parse_args())
