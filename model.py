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
    def __init__(self, in_channels=1, mid_channels=8, kernel_size=3, res="same", time_embed_dim=512, num_heads=-1):
        super().__init__()
        self.res = res
        self.in_channels = in_channels
        self.time_embed_layers = nn.Sequential(
            nn.Linear(time_embed_dim, mid_channels),
            )
        self.in_conv = nn.Sequential(Conv1d(in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
                                     nn.LeakyReLU(0.1))
        self.has_attention = num_heads > 0
        if self.has_attention:
            self.attention = AttentionBlock(mid_channels, num_heads=num_heads) 
        self.mid_conv = nn.Sequential(Conv1d(mid_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
                                 nn.LeakyReLU(0.1))
        self.out_conv = Conv1d(mid_channels, out_channels=in_channels, kernel_size=kernel_size, padding=1)
        if res == "same":
            self.op = nn.Identity()
        elif res == "down":
            self.op = max_pool_nd(1, kernel_size=2, ceil_mode=False) # 1D max pooling
        else:
            # assign F.interpolate to self.op, without calling the function
            self.op = F.interpolate


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
        if self.has_attention:
            h = self.attention(h)
        h = h + self.mid_conv(h)
        h = self.out_conv(h)
        if self.res == "up":
            y = self.op(h, scale_factor=2, mode="nearest")
        else :
            y = self.op(h)
        return y 


#%%

class Simple_Diff(nn.Module):
    """
    params:
        model_channels (int): base channel count for the model.
        levels (int): number of levels or scales of the model.
        embed_dim(int): Dimension of the encoding vector for each timestep t. Default is 128.
        proj_embed_dim (int): Dimension of the projection layer. Default is 512.

    """
    def __init__(self, params):
        super().__init__()
        set_params = list(params.keys())
        self.embed_dim = params.embed_dim if "embed_dim" in set_params else 128
        self.proj_embed_dim = params.proj_embed_dim if "proj_embed_dim" in set_params else self.embed_dim * 4
        self.diffusion_embedding = SinusoidalPositionEmbeddings(num_steps=params.num_diff_steps, dim=self.embed_dim, proj_dim=self.proj_embed_dim)
        self.levels = params.levels
        self.in_channels = params.num_coords
        self.mid_channels = params.model_channels
        # number of heads for the attention at the convolutions
        self.conv_num_heads = params.num_heads if (params.attention_at_convs) else -1
        self.blocks_highest = nn.ModuleList([ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="down", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                                             ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads)])
        self.blocks_lowest = nn.ModuleList([ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="up", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                                             ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads)])
        
        blocks = []
        for i in range(self.levels - 2):
            blocks_level = nn.ModuleList([
                ConvBlock(self.in_channels, mid_channels=self.mid_channels ,kernel_size=params.kernel_size, res="up", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="down", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads)
            ])
            blocks.append(blocks_level)
        
        self.blocks = nn.ModuleList([self.blocks_highest] + blocks + [self.blocks_lowest])
        
        # number of heads for the attention at the convolutional blocks outputs
        self.attention_at_blocks = params.attention_at_blocks
        if self.attention_at_blocks:
            self.block_num_heads = params.num_heads
            self.attention_blocks = nn.ModuleList([AttentionBlock(self.in_channels, num_heads=self.block_num_heads) for _ in range(3)])
  
    def forward(self, x_pyramid, diffusion_steps):
        """ 
        x_pyramid is a dictionary where the keys are the levels and values are a batch of interpolated trajectories at that level.
        Each sample has a shape = (1 or 3, length, num_coords) for each level
        """
        # compute the projected time embedding
        time_embed = self.diffusion_embedding(diffusion_steps)
        pred_x_pyramid = {level: None for level in range(self.levels)}
        for level in range(self.levels):
            upper_xpred = None
            down_xpred = None
            same_xpred = self.blocks[level][1](x_pyramid[level], time_embed)
            if self.attention_at_blocks:
                # apply attention to the output of the same resolution block
                same_xpred = self.attention_blocks[1](same_xpred)

            pred_x_pyramid[level] = same_xpred if pred_x_pyramid[level] is None else pred_x_pyramid[level] + same_xpred
            
            if level > 0 and level < self.levels - 1:
                # here the lower resolution is talking to/ affecting the resolution above
                # use upsampling block
                upper_xpred = self.blocks[level][0](x_pyramid[level], time_embed)
                if self.attention_at_blocks:
                    # apply attention to the output of the same resolution block
                    upper_xpred = self.attention_blocks[0](upper_xpred)

                pred_x_pyramid[level-1] = pred_x_pyramid[level-1] + upper_xpred
                # here the higher resolution is talking to/ affecting the resolution below
                # use downsampling block
                down_xpred = self.blocks[level][2](x_pyramid[level], time_embed)
                if self.attention_at_blocks:
                    # apply attention to the output of the same resolution block
                    down_xpred = self.attention_blocks[2](down_xpred)

                pred_x_pyramid[level+1] = down_xpred if pred_x_pyramid[level+1] is None else pred_x_pyramid[level+1] + down_xpred
            
            if level == 0:
                down_xpred = self.blocks[level][0](x_pyramid[level], time_embed)
                if self.attention_at_blocks:
                    # apply attention to the output of the same resolution block
                    down_xpred = self.attention_blocks[2](down_xpred)
                pred_x_pyramid[1] = down_xpred
            
            if level == self.levels - 1:
                upper_xpred = self.blocks[level][0](x_pyramid[level], time_embed)
                if self.attention_at_blocks:
                    # apply attention to the output of the same resolution block
                    upper_xpred = self.attention_blocks[0](upper_xpred)
                pred_x_pyramid[level - 1] = pred_x_pyramid[level-1] + upper_xpred
            
        return pred_x_pyramid
    

#%%

class ScIDiff(nn.Module):
  """
  params:
    model_channels (int): base channel count for the model.
    levels (int): number of levels or scales of the model.
    embed_dim(int): Dimension of the encoding vector for each timestep t. Default is 128.
    proj_embed_dim (int): Dimension of the projection layer. Default is 512.

  """
  def __init__(self, params):
    super().__init__()
    set_params = list(params.keys())
    self.embed_dim = params.embed_dim if "embed_dim" in set_params else 128
    self.proj_embed_dim = params.proj_embed_dim if "proj_embed_dim" in set_params else self.embed_dim * 4
    self.diffusion_embedding = SinusoidalPositionEmbeddings(num_steps=params.num_diff_steps, dim=self.embed_dim, proj_dim=self.proj_embed_dim)
    self.levels = params.levels
    self.in_channels = params.num_coords
    self.mid_channels = params.model_channels
    # number of heads for the attention at the convolutions
    self.conv_num_heads = params.num_heads if (params.attention_at_convs) else -1
    self.blocks = nn.ModuleList([ConvBlock(self.in_channels, mid_channels=self.mid_channels ,kernel_size=params.kernel_size, res="up", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                                 ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                                 ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="down", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads)])
    # number of heads for the attention at the convolutional blocks outputs
    self.attention_at_blocks = params.attention_at_blocks
    if self.attention_at_blocks:
      self.block_num_heads = params.num_heads
      self.attention_blocks = nn.ModuleList([AttentionBlock(self.in_channels, num_heads=self.block_num_heads) for _ in range(3)])
  
  def forward(self, x_pyramid, diffusion_steps):
    """ 
    x_pyramid is a dictionary where the keys are the levels and values are a batch of interpolated trajectories at that level.
    Each sample has a shape = (1 or 3, length, num_coords) for each level
    """
    # compute the projected time embedding
    time_embed = self.diffusion_embedding(diffusion_steps)
    pred_x_pyramid = {level: None for level in range(self.levels)}
    for level in range(self.levels):
      upper_xpred = None
      down_xpred = None
      same_xpred = self.blocks[1](x_pyramid[level], time_embed)
      if self.attention_at_blocks:
        # apply attention to the output of the same resolution block
        same_xpred = self.attention_blocks[1](same_xpred)
      pred_x_pyramid[level] = same_xpred if pred_x_pyramid[level] is None else pred_x_pyramid[level] + same_xpred
      if level > 0:
        # here the lower resolution is talking to/ affecting the resolution above
        # use upsampling block
        upper_xpred = self.blocks[0](x_pyramid[level], time_embed)
        if self.attention_at_blocks:
          # apply attention to the output of the upsampling block
          upper_xpred = self.attention_blocks[0](upper_xpred)
        pred_x_pyramid[level-1] = pred_x_pyramid[level-1] + upper_xpred
      if level < self.levels - 1:
        # here the higher resolution is talking to/ affecting the resolution below
        # use downsampling block
        down_xpred = self.blocks[2](x_pyramid[level], time_embed)
        if self.attention_at_blocks:
          # apply attention to the output of the downsampling block
          down_xpred = self.attention_blocks[2](down_xpred)
        pred_x_pyramid[level+1] = down_xpred if pred_x_pyramid[level+1] is None else pred_x_pyramid[level+1] + down_xpred
    return pred_x_pyramid

    
    
#%%


class ScIDiff_fourier(nn.Module):
  """
  params:
    model_channels (int): base channel count for the model.
    levels (int): number of levels or scales of the model.
    embed_dim(int): Dimension of the encoding vector for each timestep t. Default is 128.
    proj_embed_dim (int): Dimension of the projection layer. Default is 512.

  """
  def __init__(self, params):
    super().__init__()
    set_params = list(params.keys())
    self.embed_dim = params.embed_dim if "embed_dim" in set_params else 128
    self.proj_embed_dim = params.proj_embed_dim if "proj_embed_dim" in set_params else self.embed_dim * 4
    self.diffusion_embedding = SinusoidalPositionEmbeddings(num_steps=params.num_diff_steps, dim=self.embed_dim, proj_dim=self.proj_embed_dim)
    self.levels = params.levels
    self.in_channels = params.num_coords
    self.mid_channels = params.model_channels
    # number of heads for the attention at the convolutions
    self.conv_num_heads = params.num_heads if (params.attention_at_convs) else -1
    self.blocks = nn.ModuleList([ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                                 ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                                 ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads)])
    # number of heads for the attention at the convolutional blocks outputs
    self.attention_at_blocks = params.attention_at_blocks
    if self.attention_at_blocks:
      self.block_num_heads = params.num_heads
      self.attention_blocks = nn.ModuleList([AttentionBlock(self.in_channels, num_heads=self.block_num_heads) for _ in range(3)])
    
    self.smoother = GaussianSmoother(self.levels)
    
  
  def forward(self, x_pyramid, diffusion_steps):
    """ 
    x_pyramid is a dictionary where the keys are the levels and values are a batch of interpolated trajectories at that level.
    Each sample has a shape = (1 or 3, length, num_coords) for each level
    """
    # compute the projected time embedding
    time_embed = self.diffusion_embedding(diffusion_steps)
    pred_x_pyramid = {level: None for level in range(self.levels)}
    for level in range(self.levels):
      upper_xpred = None
      down_xpred = None
      same_xpred = self.blocks[1](x_pyramid[level], time_embed)
      if self.attention_at_blocks:
        # apply attention to the output of the same resolution block
        same_xpred = self.attention_blocks[1](same_xpred)
      pred_x_pyramid[level] = same_xpred if pred_x_pyramid[level] is None else pred_x_pyramid[level] + same_xpred
      if level > 0:
        # here the lower resolution is talking to/ affecting the resolution above
        # use upsampling block
        upper_xpred = self.blocks[0](x_pyramid[level], time_embed)
        if self.attention_at_blocks:
          # apply attention to the output of the upsampling block
          upper_xpred = self.attention_blocks[0](upper_xpred)
        pred_x_pyramid[level-1] = pred_x_pyramid[level-1] + upper_xpred
      if level < self.levels - 1:
        # here the higher resolution is talking to/ affecting the resolution below
        # use downsampling block
        down_xpred = self.blocks[2](x_pyramid[level], time_embed)
        if self.attention_at_blocks:
          # apply attention to the output of the downsampling block
          down_xpred = self.attention_blocks[2](down_xpred)
        pred_x_pyramid[level+1] = down_xpred if pred_x_pyramid[level+1] is None else pred_x_pyramid[level+1] + down_xpred
    return pred_x_pyramid


#%%


class Simple_Diff_fourier(nn.Module):
    """
    params:
        model_channels (int): base channel count for the model.
        levels (int): number of levels or scales of the model.
        embed_dim(int): Dimension of the encoding vector for each timestep t. Default is 128.
        proj_embed_dim (int): Dimension of the projection layer. Default is 512.

    """
    def __init__(self, params):
        super().__init__()
        set_params = list(params.keys())
        self.embed_dim = params.embed_dim if "embed_dim" in set_params else 128
        self.proj_embed_dim = params.proj_embed_dim if "proj_embed_dim" in set_params else self.embed_dim * 4
        self.diffusion_embedding = SinusoidalPositionEmbeddings(num_steps=params.num_diff_steps, dim=self.embed_dim, proj_dim=self.proj_embed_dim)
        self.levels = params.levels
        self.in_channels = params.num_coords
        self.mid_channels = params.model_channels
        # number of heads for the attention at the convolutions
        self.conv_num_heads = params.num_heads if (params.attention_at_convs) else -1
        self.blocks_highest = nn.ModuleList([ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                                             ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads)])
        self.blocks_lowest = nn.ModuleList([ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                                             ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads)])
        
        blocks = []
        for i in range(self.levels - 2):
            blocks_level = nn.ModuleList([
                ConvBlock(self.in_channels, mid_channels=self.mid_channels ,kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads),
                ConvBlock(self.in_channels, mid_channels=self.mid_channels, kernel_size=params.kernel_size, res="same", time_embed_dim=self.proj_embed_dim, num_heads=self.conv_num_heads)
            ])
            blocks.append(blocks_level)
        
        self.blocks = nn.ModuleList([self.blocks_highest] + blocks + [self.blocks_lowest])
        
        # number of heads for the attention at the convolutional blocks outputs
        self.attention_at_blocks = params.attention_at_blocks
        if self.attention_at_blocks:
            self.block_num_heads = params.num_heads
            self.attention_blocks = nn.ModuleList([AttentionBlock(self.in_channels, num_heads=self.block_num_heads) for _ in range(3)])
  
    def forward(self, x_pyramid, diffusion_steps):
        """ 
        x_pyramid is a dictionary where the keys are the levels and values are a batch of interpolated trajectories at that level.
        Each sample has a shape = (1 or 3, length, num_coords) for each level
        """
        # compute the projected time embedding
        time_embed = self.diffusion_embedding(diffusion_steps)
        pred_x_pyramid = {level: None for level in range(self.levels)}
        for level in range(self.levels):
            upper_xpred = None
            down_xpred = None
            same_xpred = self.blocks[level][1](x_pyramid[level], time_embed)
            if self.attention_at_blocks:
                # apply attention to the output of the same resolution block
                same_xpred = self.attention_blocks[1](same_xpred)

            pred_x_pyramid[level] = same_xpred if pred_x_pyramid[level] is None else pred_x_pyramid[level] + same_xpred
            
            if level > 0 and level < self.levels - 1:
                # here the lower resolution is talking to/ affecting the resolution above
                # use upsampling block
                upper_xpred = self.blocks[level][0](x_pyramid[level], time_embed)
                if self.attention_at_blocks:
                    # apply attention to the output of the same resolution block
                    upper_xpred = self.attention_blocks[0](upper_xpred)

                pred_x_pyramid[level-1] = pred_x_pyramid[level-1] + upper_xpred
                # here the higher resolution is talking to/ affecting the resolution below
                # use downsampling block
                down_xpred = self.blocks[level][2](x_pyramid[level], time_embed)
                if self.attention_at_blocks:
                    # apply attention to the output of the same resolution block
                    down_xpred = self.attention_blocks[2](down_xpred)

                pred_x_pyramid[level+1] = down_xpred if pred_x_pyramid[level+1] is None else pred_x_pyramid[level+1] + down_xpred
            
            if level == 0:
                down_xpred = self.blocks[level][0](x_pyramid[level], time_embed)
                if self.attention_at_blocks:
                    # apply attention to the output of the same resolution block
                    down_xpred = self.attention_blocks[2](down_xpred)
                pred_x_pyramid[1] = down_xpred
            
            if level == self.levels - 1:
                upper_xpred = self.blocks[level][0](x_pyramid[level], time_embed)
                if self.attention_at_blocks:
                    # apply attention to the output of the same resolution block
                    upper_xpred = self.attention_blocks[0](upper_xpred)
                pred_x_pyramid[level - 1] = pred_x_pyramid[level-1] + upper_xpred
            
        return pred_x_pyramid
    
    
    
    
    
