"""
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from torch import nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
import logger
import torch

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def mse_loss(x_hat, x):
    """
    Function that computes the MSE loss between two tensors.
    Args:
        x_hat (torch.Tensor): predicted tensor of shape (batch_size, coords, length), e.g:
        (64,3,2048).
    """
    # compute the squared error without reduction
    se = (x-x_hat) ** 2
    # average over the channels and length
    mse = mean_flat(se)
    # average over the batches
    return mse.mean()

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def max_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D max pooling module.
    """
    if dims == 1:
        return nn.MaxPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.MaxPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.MaxPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def interpolate_nscales(sample, scales=2, method="nearest", to_numpy=False):
    '''
    Function that applies a pyramidal interpolation to a batch of samples. A pyramidal interpolation
    is a sequence of interpolations with a scale factor of 1/2, 1/4, 1/8 and so on.
    The interpolation method can be chosen among the ones available in PyTorch.

    Args:
        sample (torch.Tensor): a tensor of shape (batch_size, coords, length).
        scales (int): number of scaling factors to apply to the interpolation.
        method (str): interpolation method. Default is 'nearest-exact'.
        to_numpy (bool): if True, the output will be a dictionary with numpy arrays, otherwise they will
        be Tensors. Default is False.

    Returns:
        pyramidal_sample (dict): a dictionary with keys = levels and values = interpolated samples
     
    '''
    # sample has to be a Tensor of shape (batch_size, coords, length)
    # e.g for one trajectory, Tensor should be (1, 3, 2000)
    logger.log(sample.shape)
    pyramidal_sample = {0: sample}
    for i in range(1, scales):
        scale = 1/2**i
        y = F.interpolate(sample, scale_factor=scale, mode=method)
        if(to_numpy):
            # we have to permute the dimensions to plot them
            # delete the batch dimension
            # and trasform to numpy
            pyramidal_sample[i] = y.squeeze(0).permute(1,0).numpy()
        else:
            pyramidal_sample[i] = y
    return pyramidal_sample



def fourier_nscales(sample, scales = 2, to_numpy = False):
    if torch.tensor(sample):
        pyramidal_sample = {0: sample}
    else:
        pyramidal_sample = {0: torch.Tensor(sample)}
    stds = [1, 2, 4, 16, 32, 64, 128, 256, 512]
    for i in range(1, scales):
        std = stds[i-1]
        
        if torch.is_tensor(sample):
            y = sample.numpy()
        else:
            y = sample
            
        y = gaussian_filter1d(y, std, mode = "constant", cval = 0.0)
        
        if to_numpy:
            pyramidal_sample[i] = y
        else:
            pyramidal_sample[i] = torch.Tensor(y)
    
    return pyramidal_sample



def _nested_map(struct, map_fn):
  '''
  With map_fn = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
  this function will dive into an structure until it finds a tensor, and then
  send it to a device.
  Example:
  if struct is a dict like:
  x = {"audio": Tensor(64,22000),
   "spectrogram": Tensor(64,1024,128)}
  then the result is
  x = {"audio": Tensor(64,22000).to(device),
    "spectrogram": Tensor(64,1024,128).to(device)}
  '''
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)
