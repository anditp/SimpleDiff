import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from attention import AttentionBlock
import os
from diffusion import *
import torch.distributed as dist
import logger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import mse_loss, _nested_map
from tqdm import tqdm


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
    def __init__(self, in_channels=1, mid_channels=8, out_channels = 1, kernel_size=3, time_embed_dim=512, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_layers = nn.Sequential(
            nn.Linear(time_embed_dim, mid_channels),
            )
        self.in_conv = nn.Sequential(Conv1d(in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding="same"),
                                     nn.BatchNorm1d(mid_channels),
                                     nn.LeakyReLU(0.1),
                                     nn.Dropout(0.1))
        self.mid_conv = nn.Sequential(Conv1d(mid_channels, out_channels=mid_channels, kernel_size=kernel_size, padding="same"),
                                 nn.BatchNorm1d(mid_channels),
                                 nn.LeakyReLU(0.1),
                                 nn.Dropout(0.1))
        self.out_conv = Conv1d(mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same")


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
        return h



#%%

class ScI_MR(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        set_params = list(params.keys())
        self.embed_dim = params.embed_dim if "embed_dim" in set_params else 128
        self.proj_embed_dim = params.proj_embed_dim if "proj_embed_dim" in set_params else self.embed_dim * 4
        self.diffusion_embedding = SinusoidalPositionEmbeddings(num_steps=params.num_diff_steps, dim=self.embed_dim, proj_dim=self.proj_embed_dim)
        self.in_channels = params.num_coords
        self.mid_channels = params.model_channels
        self.relu = nn.LeakyReLU(0.1)
        
        self.condition_preprocess = ConvBlock(self.in_channels, mid_channels=self.mid_channels, out_channels = self.mid_channels, kernel_size=params.kernel_size, time_embed_dim=self.proj_embed_dim)
        
        self.level_preprocess = ConvBlock(self.in_channels, mid_channels=self.mid_channels, out_channels = self.mid_channels, kernel_size=params.kernel_size, time_embed_dim=self.proj_embed_dim)

        self.conditioned_network_block1 = ConvBlock(2 * self.mid_channels, mid_channels = 2 * self.mid_channels, out_channels = self.mid_channels, kernel_size=params.kernel_size, time_embed_dim=self.proj_embed_dim)
        self.conditioned_network_block2 = ConvBlock(self.mid_channels, mid_channels = self.mid_channels, out_channels = self.in_channels, kernel_size=params.kernel_size, time_embed_dim=self.proj_embed_dim)
    
    
    def forward(self, x, t, c):
        t = self.diffusion_embedding(t)
        d = self.condition_preprocess(c, t)
        d = self.relu(d)
        h = self.level_preprocess(x, t)
        h = self.relu(h)
        h = torch.concat((h, d), dim = 1)
        h = self.conditioned_network_block1(h, t)
        h = self.relu(h)
        h = self.conditioned_network_block2(h, t)
        return h




#%%


class MR_Learner:
    def __init__(self, model_dir, model, dataset, optimizer, params, **kwargs):
      os.makedirs(model_dir, exist_ok=True)
      self.model_dir = model_dir
      self.model = model
      self.dataset = dataset
      self.optimizer = optimizer
      self.params = params
      self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get("fp16", False))
      self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get("fp16", False))
      self.step = 0
      self.checkpoints_hop = kwargs.get("checkpoints_hop", 50000)
      self.summary_hop = kwargs.get("summary_hop", 512)
      # build diffusion process with a given schedule
      betas = create_beta_schedule(steps=self.params.num_diff_steps, scheduler=self.params.scheduler)
      self.diffuser = GaussianDiffusion(betas)
      self.loss_fn = nn.MSELoss(reduction='mean')
      self.summary_writer = None
      self.max_grad_norm = kwargs.get("max_grad_norm", 1)

    def state_dict(self):
      if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
        model_state = self.model.module.state_dict()
      else:
        model_state = self.model.state_dict()
      return {
          'step': self.step,
          'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
          'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
          'params': dict(self.params),
          'scaler': self.scaler.state_dict(),
      }

    def load_state_dict(self, state_dict):
      if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
        self.model.module.load_state_dict(state_dict['model'])
      else:
        self.model.load_state_dict(state_dict['model'])
      self.optimizer.load_state_dict(state_dict['optimizer'])
      self.scaler.load_state_dict(state_dict['scaler'])
      self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
      save_basename = f'{filename}-{self.step}.pt'
      save_name = f'{self.model_dir}/{save_basename}'
      link_name = f'{self.model_dir}/{filename}.pt'
      torch.save(self.state_dict(), save_name)
      if os.path.islink(link_name):
          os.unlink(link_name)
      os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
      try:
        checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
        self.load_state_dict(checkpoint)
        return True
      except FileNotFoundError:
        return False

    def train(self, max_steps=None):
      device = next(self.model.parameters()).device
      while True:
        # number of epochs = max_steps / num_batches
        # e.g. for max_steps = 100000 and num_batches = 1000, we have 100 epochs
        for features in self.dataset:
            logger.log(f'Epoch {self.step // 327160}')
            if max_steps is not None and self.step >= max_steps:
                # Save final checkpoint.
                self.save_to_checkpoint()
                return
            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            loss = self.train_step(features)
            if torch.isnan(loss).any():
                raise RuntimeError(f'Detected NaN loss at step {self.step}.')
            if self.is_master:
                if self.step % self.summary_hop == 0:
                  self._write_summary(self.step, loss)
                if self.step % self.checkpoints_hop == 0:
                  self.save_to_checkpoint()
            self.step += 1
              
            torch.cuda.empty_cache()

    def train_step(self, features):
      for param in self.model.parameters():
        param.grad = None

      device = features[0].device # device of the batch
      B = features[0].shape[0] # batch size
      loss_acum = 0
      loss = torch.zeros(1, device = device)
      
      for level in range(self.params.levels - 1, -1, -1):
          if level == self.params.levels - 1:
              condition = torch.zeros_like(features[0])
          else:
              condition = features[level + 1]
          with self.autocast:
            # create a tensor with random diffusion times for each sample in the batch
            diff_steps = torch.randint(0, self.params.num_diff_steps, (B,), device=device)
            # diffusion process
            noisy_batch, noise = self.diffuser.forward_diffusion_process(features[level], diff_steps, device=device)
            # forward pass
            # predicted is also a dictionary with the same structure of noisy_batch and features
            predicted = self.model(noisy_batch, diff_steps, condition)
            # compute loss
            loss += self.loss_fn(noise, predicted)
            loss_acum += loss.item()
    
      # backward pass with scaling to avoid underflow gradients
      self.scaler.scale(loss).backward()
      # unscale the gradients before clipping them
      self.scaler.unscale_(self.optimizer)
      # clip gradients
      self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
      # update optimizer
      self.scaler.step(self.optimizer)
      self.scaler.update()

      return loss_acum / self.params.levels

    def _write_summary(self, step, loss):
      """
      Function that adds to Tensorboard the loss and the gradient norm.
      """
      writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
      writer.add_scalar('train/loss', loss, step)
      writer.add_scalar('train/grad_norm', self.grad_norm, step)
      writer.flush()
      self.summary_writer = writer







        
        



    
    
    
    
    
    
    
    