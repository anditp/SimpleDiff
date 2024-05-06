import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose



class LagrDataset(Dataset):
    
    def __init__(self, npy_filename, root_dir = None, transform = None):
        self.npy_filename = npy_filename
        self.root_dir = root_dir
        self.transform = transform
        
        self.data = self.transform(np.load(npy_filename, allow_pickle = True, mmap_mode = "r"))
    
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx]


class ToTensor(object):
    
    def __call__(self, sample):
        return torch.from_numpy(sample).float()



def dataset_from_file(npy_filename, batch_size):
    
    transform = [ToTensor()]
    dataset = LagrDataset(npy_filename, transform = transform)
    
    return torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            pin_memory = True,
            drop_last = True)



