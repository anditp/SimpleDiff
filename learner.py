import os
import torch
from utils import mse_loss, _nested_map
from torch import nn
from tqdm import tqdm
from diffusion import *
import torch.distributed as dist
import logger

class ScIDiffLearner:
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
      features = next(self.dataset)
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

    with self.autocast:
      # create a tensor with random diffusion times for each sample in the batch
      diff_steps = torch.randint(0, self.params.num_diff_steps, (B,), device=device)
      # diffusion process
      noisy_batch, noise = self.diffuser.forward_diffusion_process_dict(features, diff_steps, device=device)
      # forward pass
      # predicted is also a dictionary with the same structure of noisy_batch and features
      predicted = self.model(noisy_batch, diff_steps)
      # compute loss
      loss = self.compute_loss(noise, predicted)

    # backward pass with scaling to avoid underflow gradients
    self.scaler.scale(loss).backward()
    # unscale the gradients before clipping them
    self.scaler.unscale_(self.optimizer)
    # clip gradients
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
    # update optimizer
    self.scaler.step(self.optimizer)
    self.scaler.update()

    return loss

  def _write_summary(self, step, loss):
    """
    Function that adds to Tensorboard the loss and the gradient norm.
    """
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


  def compute_loss(self, true_vals, predictions):
    """
    Function that computes the loss in a batch. The loss is first computed at each level and then averaged
    by the levels.

    Args:
        true_vals (dict): dictionary with the true values of the batch. Shape is (batch_size, coords, length).
        predictions (dict): dictionary with the predicted values of the batch.
    
    Returns:
        torch.float32: the loss value.
    """
    # compute the loss at the highest level
    loss_accum = self.loss_fn(predictions[0], true_vals[0])
    levels = self.params.levels
    for level in range(1, levels):
      loss_val = self.loss_fn(predictions[level], true_vals[level])
      loss_accum += loss_val
    return loss_accum/levels
