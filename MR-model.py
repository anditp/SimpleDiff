from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from attention import AttentionBlock


def Conv1d(*args, **kwargs):
    """
    Conv1d layer with default initialization.
    """
    layer = nn.Conv1d(*args, **kwargs)
    kaiming_normal_(layer.weight)
    return layer

class SinusoidalPositionEmbeddings(nn.Module):
    """Block that builds the sinusoidal position embeddings, as described in
        "DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS".The embeddings
        are passed through two linear layers with SiLu activations.
        
        Args:
            num_steps (int): Number of steps in the diffusion process (T in the paper).
            dim (int): Dimension of the encoding vector for each timestep t.
            proj_dim (int): Dimension of the projection layer.
    """
    def __init__(self, num_steps, dim, proj_dim):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(num_steps,dim), persistent=False)
        self.projection1 = nn.Sequential(nn.Linear(dim, proj_dim), nn.SiLU())  # [1,512]
        self.projection2 = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.SiLU()) 

    def forward(self, diffusion_step):
        """
        The forward pass of the sinusoidal position embeddings. It takes as input
        the diffusion steps (shape=(steps,)) and returns the corresponding projected embeddings.
        """
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = self.projection2(x)
        return x

    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)  # [T,1]
        dim_tensor = torch.arange(dim//2).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dim_tensor * 4.0 / (dim//2 -1.0))     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=-1) # [T,128]
        return table
  
    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)



class ConvBlock(nn.Module):
    """
    Block of convolutions, that process one sample at one scale. It also process the timestep
    embedding. 

      :param in_channels: number of input channels; corresponds to the number of coordinates.
      :param mid_channels: number of channels in the hidden layers.
      :param kernel_size: kernel size of the convolutional layers.
      :param res: indicates if a scaling should be performed at the end. Can be "same" (same dimensions), 
      'down' (downscaling to half) or 'up' (upscaling to double).
      :param time_embed_dim: dimension of the time embedding. Default is 512.
      :param num_heads: number of heads for the multi-head attention. Default is -1, which means no attention.
    """
    def __init__(self, in_channels=1, mid_channels=8, kernel_size=3, time_embed_dim=512, **kwargs):
        super().__init__()
        self.res = res
        self.in_channels = in_channels
        self.time_embed_layers = nn.Sequential(
            nn.Linear(time_embed_dim, mid_channels),
            )
        self.in_conv = nn.Sequential(Conv1d(in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
                                     nn.BatchNorm1d(mid_channels),
                                     nn.LeakyReLU(0.1),
                                     nn.Dropout(0.1))
        self.mid_conv = nn.Sequential(Conv1d(mid_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
                                 nn.BatchNorm1d(mid_channels),
                                 nn.LeakyReLU(0.1),
                                 nn.Dropout(0.1))
        self.out_conv = Conv1d(mid_channels, out_channels=in_channels, kernel_size=kernel_size, padding=1)


    def forward(self, x, time_embed):
        """
        Args:
            x (torch.Tensor): input of the block. x.shape should be (N, 1, L) where N is the batch size and L is the length of the sample
            or (N, 3, L) for 3D samples. Note that the channel dimension must come before the length dimension.
            time_embed (torch.Tensor): time embedding of the sample. time_embed.shape should be (N, time_embed_dim)
        Returns:
            y (torch.Tensor): output of the block, with the same shape as x.
        """

        h = self.in_conv(x) # (batch,coordinates, 2048 or less)
        time_embed = self.time_embed_layers(time_embed).type(h.dtype) # (batch,8)
        while len(time_embed.shape) < len(h.shape):
            time_embed = time_embed[..., None] # (batch,8,1)
        h = h + time_embed
        h = self.mid_conv(h)
        h = self.out_conv(h)
        return y 
    
    
    
    
    
    
    
    