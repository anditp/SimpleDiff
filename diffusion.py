import torch
from utils import *
import torch.nn.functional as F
import math
from utils import fourier_nscales

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    :return: a tensor of betas.
    Taken from https://github.com/SmartTURB/diffusion-lagr/blob/master/guided_diffusion/gaussian_diffusion.py
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Function that returns a linear schedule of beta values for the diffusion process.
    Args:
        timesteps (int): Number of timesteps.
        beta_start (float): Starting beta value.
        beta_end (float): Ending beta value.

    Returns:
        torch.Tensor: Beta values for the diffusion process.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def tanh61_beta_schedule(timesteps, t0=6, t1=1):
    """
    tanh6-1 schedule
    """
    return betas_for_alpha_bar(
                timesteps,
                lambda t: -math.tanh((t0 + t1) * t - t0) + math.tanh(t1),
            )

def create_beta_schedule(steps, scheduler="linear", **kwargs):
    """
    Function that returns a beta schedule for the diffusion process.
    Args:
        steps (int): Number of timesteps.
        scheduler (str): Scheduler to use for the diffusion process.
        kwargs: Additional arguments for the scheduler.

    Returns:
        torch.Tensor: Beta values for the diffusion process.
    """
    if(scheduler == "cosine"):
      betas = cosine_beta_schedule(timesteps=steps, **kwargs)
    elif(scheduler == "tanh61"):
      betas = tanh61_beta_schedule(timesteps=steps, **kwargs)
    else:
      betas = linear_beta_schedule(timesteps=steps, **kwargs)
    return betas

class GaussianDiffusion:
    """
    Class for Gaussian diffusion process.
    
    Args:
        betas (torch.Tensor): Beta values for the diffusion process.
    """
    def __init__(self, betas) -> None:
        self.betas = betas
        self.alphas = 1. - betas
        # accumulated product of alphas for each time step (1,.., T)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # and until time t-1 (1,.., t-1)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # terms with square root of all the accumulated alphas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # variance of the posterior q(x_{t-1} | x_t, x_0)
        # according to eq. 7 in Ho et al. (2020)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        #
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def forward_diffusion_process_dict(self, x_0, t, fourier = False, device="cpu"):
        """ 
        Takes a dictionary of batches and timesteps as input and returns the noisy version of it.
        Args:
            x_0 (dict): Dictionary of batches.
            t (torch.Tensor): Timesteps.
        Returns:
            dict(torch.Tensor): Noisy data.
            dict(torch.Tensor): Noise added to each level.
        """
        noises = {}
        
        if fourier:
            levels = len(x_0)
            noise = np.random.randn_like(x_0)
            pyramidal_noise = fourier_nscales(noise, scales = levels)
            pyramidal_noise = _nested_map(pyramidal_noise, lambda x: x.to(device))
            for level, trajectory in x_0.items():
                noisy_traj, noise = self.forward_diffusion_process(trajectory, t, pyramidal_noise[levels])
                x_0[level] = noisy_traj.to(device)
                noises[level] = noise.to(device)
        else:
            for level, trajectory in x_0.items():
                noisy_traj, noise = self.forward_diffusion_process(trajectory, t)
                x_0[level] = noisy_traj.to(device)
                noises[level] = noise.to(device)
        return x_0, noises
    
    def forward_diffusion_process(self, x_0, t, noise = None):
        """ 
        Takes a data point (or a batch) and a timestep (or batch of timesteps) 
        as input and returns the noisy version of it.

        Returns:
            torch.Tensor: Noisy data.
            torch.Tensor: Noise added to the data.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t * x_0 \
        + sqrt_one_minus_alphas_cumprod_t * noise, noise
        
        
        
        
        
        
        
    
    
