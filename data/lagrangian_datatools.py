import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
sys.path.append('../')
from utils import interpolate_nscales, fourier_nscales, GaussianSmoother
import logger


# interpolation methods
def interp1d(sample, dim):
    original_length = sample.shape[0]
    # time vector [1,..., 2000]
    T = np.linspace(1, original_length, num=original_length)
    xnew = np.linspace(1, original_length, num=dim)
    interpolated = np.interp(xnew, T, sample)
    return interpolated

class ParticleDataset(Dataset):
    def __init__(self, npy_filename, root_dir=None, transform=None):
        super().__init__()
        self.npy_filename = npy_filename
        self.root_dir = root_dir
        self.transform = transform
        self.data = np.load(self.npy_filename, encoding="ASCII", allow_pickle=True, mmap_mode='r')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        part_traj = self.data[idx]
        if self.transform:
            part_traj = self.transform(part_traj)

        return part_traj


class NoiseDataset(Dataset):
    def __init__(self, shape, transform = None):
        super().__init__()
        self.transform = transform
        self.shape = shape
        self.data = torch.randn(*shape)
    
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        if self.transform:
            traj = self.data[idx]
        
        return traj


class ParticleDatasetVx(Dataset):
    def __init__(self, npy_filename, root_dir, transform=None):

        self.npy_filename = npy_filename
        self.root_dir = root_dir
        self.transform = transform
        self.data = np.load(self.npy_filename, encoding="ASCII", allow_pickle=True, mmap_mode='r+')[:,:,0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        part_traj = self.data[idx, :]
        if self.transform:
            part_traj = self.transform(part_traj)

        return part_traj

class StandardScaler(object):
    """Standardize the data"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std

class MinMaxScaler(object):
    """Standardize the data and sets it to [-1,1] range"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, sample):
        return 2*(sample - self.min) / (self.max - self.min) -1
    
class Interpolation(object):
    def __init__(self, dim, scale_factor=2, method='linear'):
        self.dim = dim
        self.scale_factor = scale_factor
        self.method = method

    def __call__(self, sample):
        if(self.dim==3):
            y = F.interpolate(sample.unsqueeze(-1), self.scale_factor, mode=self.method)
        else:
            y = sample
        return y

class TakeOneCoord(object):
    """Take one coordinate from the trajectory. The object must be a Tensor that
    has shape (num_coords, length). Therefore, take note on this. If you're working
    with the npy file, you have to permute the dimensions fist with TensorChanFirst().
    
    Args:
        coord (int): coordinate to take from the trajectory.
        
    Returns:
        torch.Tensor: a tensor of shape (1, coord, length) with the trajectory of the chosen coordinate.
    """
    def __init__(self, coord):
        self.coord = coord

    def __call__(self, sample):
        traj = sample[self.coord, :]
        return  traj.unsqueeze(0)

class TensorChanFirst(object):
    def __init(self):
        pass
    def __call__(self, sample):
        return sample.permute(1,0)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # return scalar type float32
        # which is the default type for model weights
        return torch.from_numpy(sample).float()


class Collator:
    def __init__(self, levels):
        # levels is the number of scales for the interpolation
        self.levels = levels
        pass

    def collate(self, minibatch):
        """
        Function that takes a list of records and returns a dictionary with the
        batch interpolated to the different scales.

        Args:
            minibatch (list): list of records. Each record is a tensor of shape (num_coords, 2000).
        
        Returns:
            dict[int]: a dictionary with each scale as key and the batch interpolated to the 
            corresponding scale.
        """        

        trajectories = torch.stack(minibatch, dim=0)
        # get a dictionary with the batch rescaled to the different levels
        batch_interp = interpolate_nscales(trajectories, scales=self.levels)
        return batch_interp



class Collator_fourier:
    def __init__(self, levels):
        # levels is the number of scales for the interpolation
        self.levels = levels
        pass

    def collate(self, minibatch):
        """
        Function that takes a list of records and returns a dictionary with the
        batch interpolated to the different scales.

        Args:
            minibatch (list): list of records. Each record is a tensor of shape (num_coords, 2000).
        
        Returns:
            dict[int]: a dictionary with each scale as key and the batch interpolated to the 
            corresponding scale.
        """        

        trajectories = torch.stack(minibatch, dim=0)
        # get a dictionary with the batch rescaled to the different levels
        batch_fourier = fourier_nscales(trajectories, scales=self.levels, smoother = GaussianSmoother(self.levels))
        return batch_fourier

    

class ToDictTensor(object):
    """Converts ndarray to a dictionary of Tensors."""

    def __call__(self, sample):
        trajectory = sample

        return {
            'audio': torch.from_numpy(trajectory).float(),
            'spectrogram': None
        }


def dataset_from_file(npy_fname, 
                      batch_size, 
                      levels, 
                      coordinate=None, 
                      is_distributed=False, 
                      fourier = False, **kwargs):
    """
    Function that returns a DataLoader for the Lagrangian trajectories dataset.

    Args:

        npy_fname (str): path to the .npy file containing the dataset
        batch_size (int): batch size.
        levels (int): number of levels to use for the multiscale interpolation.
        is_distributed (bool): whether to use a distributed sampler or not.
        **kwargs: additional arguments to pass to the DataLoader constructor.

    Returns:
        torch.utils.data.DataLoader: a DataLoader for the dataset.
    """
    # read 3D trajectories
    # usual transformations are ToTensor() and permute(1, 0)
    # to get channel-first tensors
    transforms = [ToTensor(), TensorChanFirst()]
    if coordinate is not None:
        transforms.append(TakeOneCoord(coord=coordinate))
    dataset = ParticleDataset(npy_fname, transform=Compose(transforms))
    
    col = Collator_fourier(levels = levels)
    
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn= col.collate,
        shuffle=not is_distributed,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        drop_last=True,
        **kwargs)
    
    

    
    
