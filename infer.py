import torch
from argparse import ArgumentParser
from model import ScIDiff, Simple_Diff, ScIDiff_fourier, Simple_Diff_fourier
from MR_model import ScI_MR
import yaml
from attrdict import AttrDict
from diffusion import create_beta_schedule
import numpy as np
from utils import interpolate_nscales, fourier_nscales, _nested_map
from torch.nn import functional as F
import logger

MIN_VALS = np.array([-9.97037474, -8.63455392, -8.3230226 ])
MAX_VALS = np.array([ 9.78241835, 10.2621928,   9.73699859])

def reverse_minmax_norm(x, coordinate = -1, from_numpy=False):
    """
    Function that performs reverse min-max normalization on a tensor.
    Args:
        x: tensor of shape (N, length, num_coords)
        coordinate: if -1, the tensor is assumed to be of shape (N, 3, length), otherwise (N, 1, length). 
        We need this value to know which min-max values to use.
        from_numpy: if True, the input tensor is a numpy array
    """
    if coordinate == -1:
        if from_numpy:
            return (x + 1) * (MAX_VALS - MIN_VALS) / 2 + MIN_VALS
        return (x + 1) * (torch.tensor(MAX_VALS, device=x.device) - torch.tensor(MIN_VALS, device=x.device)) / 2 + torch.tensor(MIN_VALS, device=x.device)
    else:
        if from_numpy:
            return (x + 1) * (MAX_VALS[coordinate] - MIN_VALS[coordinate]) / 2 + MIN_VALS[coordinate]
        return (x + 1) * (torch.tensor(MAX_VALS[coordinate], device=x.device) - torch.tensor(MIN_VALS[coordinate], device=x.device)) / 2 + torch.tensor(MIN_VALS[coordinate], device=x.device)


def generate_trajectories_mr(args, model, model_params, device):
    
    
    with torch.no_grad():
        # get noise schedule
        training_noise_schedule = create_beta_schedule(steps=model_params.num_diff_steps, scheduler=model_params.scheduler).numpy()
        inference_noise_schedule = training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        B = model_params.batch_size
        """ 
        T = []
        # compute the aligned diffusion steps for sampling
        # this is only relevant if we use the fast sampling procedure
        for s in range(len(inference_noise_schedule)):
            # eq 14 of the DiffWave paper, appendix B
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                break
        
        T = np.array(T, dtype=np.float32)   
        """
        # get random noise vector at several scales
        # random tensor must be of shape (N, num_coords, length + padding)
        x_0 = np.random.randn(B, model_params.num_coords, model_params.traj_len * model_params.levels)
        trajectories = np.split(x_0, model_params.levels, axis = -1)
        gen_x = {}
        for level in range(model_params.levels - 1, -1, -1):
            noise = trajectories[level]
            gen_x[level] = torch.Tensor(noise).to(device)
            
            if level == model_params.levels - 1:
                condition = torch.zeros_like(noise)
            else:
                condition = gen_x[level + 1]
            # T-1 steps of denoising
            # we are iterating backwards
            for t in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[t]**0.5
                c2 = beta[t] / (1 - alpha_cum[t])**0.5
                pred_noise = model(gen_x[level], torch.tensor([t], device=device), condition)
                # denoise
                gen_x[level] = c1 * (gen_x[level] - c2 * pred_noise)
                if(t > 0):
                    noise = torch.randn_like(gen_x[level]).to(device)
                    sigma = ((1.0 - alpha_cum[t-1]) / (1.0 - alpha_cum[t]) * beta[t])**0.5
                    #sigma = beta[t-1]**0.5
                    gen_x[level] += sigma * noise
                    gen_x[level] = gen_x[level].clamp(-1.0, 1.0)

    return gen_x[0]

def generate_trajectories(args, model, model_params, device, fast_sampling=False):
    """
    Function that generates trajectories from a trained model. It follows Algorithm 2 of the 
    Denoising Diffusion Probabilistic Models paper.
    """

    with torch.no_grad():
        # get noise schedule
        training_noise_schedule = create_beta_schedule(steps=model_params.num_diff_steps, scheduler=model_params.scheduler).numpy()
        inference_noise_schedule = np.array(model_params.inference_noise_schedule) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        B = model_params.batch_size
        """ 
        T = []
        # compute the aligned diffusion steps for sampling
        # this is only relevant if we use the fast sampling procedure
        for s in range(len(inference_noise_schedule)):
            # eq 14 of the DiffWave paper, appendix B
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                break
        
        T = np.array(T, dtype=np.float32)   
        """
        # get random noise vector at several scales
        # random tensor must be of shape (N, num_coords, length + padding)
        x_0 = np.random.randn(B, model_params.num_coords, model_params.traj_len * model_params.levels)
        #if model_params.type in ["fourier", "simple_fourier"]:
         #   gen_x = fourier_nscales(x_0, scales = model_params.levels)
          #  gen_x = _nested_map(gen_x, lambda x: x.to(device))
        #else:
         #   gen_x = interpolate_nscales(x_0, scales=model_params.levels)
        trajectories = np.split(x_0, model_params.levels, axis = -1)
        gen_x = {}
        for level, noise in enumerate(trajectories):
            gen_x[level] = torch.Tensor(noise).to(device)
        # T-1 steps of denoising
        # we are iterating backwards
        for t in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[t]**0.5
            c2 = beta[t] / (1 - alpha_cum[t])**0.5
            pred_noise = model(gen_x, torch.tensor([t], device=device))
            # denoise at every scale
            for level, v in pred_noise.items():
                gen_x[level] = c1 * (gen_x[level] - c2 * v)
                if(t > 0):
                    noise = torch.randn_like(gen_x[level]).to(device)
                    sigma = ((1.0 - alpha_cum[t-1]) / (1.0 - alpha_cum[t]) * beta[t])**0.5
                    #sigma = beta[t-1]**0.5
                    gen_x[level] += sigma * noise
                    gen_x[level] = gen_x[level].clamp(-1.0, 1.0)

        gen_full_scale = gen_x[0]
        # remove padding at the last dimension (length)
        #gen_full_scale = gen_full_scale[:, :, 24:-24]
        gen_full = torch.zeros_like(gen_x[0])
        for level in gen_x.keys():
            gen_full += gen_x[level]
    return gen_x, gen_full



def main(args):
    logger.configure(dir = "logs")
    
    # load model params file
    with open(args.params_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader) 
        cfg = AttrDict(config)
    model_params = cfg.parameters
    assert model_params.coordinate >= -1 and model_params.coordinate <=2 and type(model_params.coordinate) == int
    model_params.coordinate = None if model_params.coordinate==-1 else model_params.coordinate
    model_params.num_coords = 3 if model_params.coordinate is None else 1
    # set device for inference operations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = args.num_samples
    B = model_params.batch_size
    # generate trajectories, possible to use batches
    batches_gen = []
    
    # By default we load the weights.pt file from the model directory.
    chck_path = f"{args.model_dir}/weights.pt"
    checkpoint = torch.load(chck_path, map_location=device)
    if model_params.type == "fourier":
        model = ScIDiff_fourier(model_params).to(device=device)
    elif model_params.type == "simple":
        model = Simple_Diff(model_params).to(device=device)
    elif model_params.type == "simple_fourier":
        model = Simple_Diff_fourier(model_params).to(device=device)
    elif model_params.type == "sci_mr":
        model = ScI_MR(model_params).to(device=device)
    else:
        model = ScIDiff(model_params).to(device=device)
    model.load_state_dict(checkpoint["model"]) # if the params settings do not match with the checkpoint, this will fail
    model.eval()
    
    
    for _ in range(N//B):
        logger.log("Iteration %d \n" % _)
        if model_params.type == "sci_mr":
            gen_full = generate_trajectories_mr(args, model, model_params, device)
        else:
            gen_samples, gen_full = generate_trajectories(args, model, model_params, device, fast_sampling=args.fast)
        batches_gen.append(gen_full)
    # concatenate batches in a single one
    gen_samples = torch.cat(batches_gen, dim=0)
    # permute to (N, length, num_coords)
    gen_samples = gen_samples.permute(0, 2, 1)
    # transform to numpy array
    gen_samples = gen_samples.cpu().numpy()
    logger.log(gen_samples.shape)
    # perform reverse normalization
    if args.normalized:
        gen_samples = reverse_minmax_norm(gen_samples, coordinate=model_params.coordinate, from_numpy=True)
    # save to file
    with open(args.output, "wb") as f:
      np.save(f, gen_samples)
  
if __name__ == "__main__":
  parser = ArgumentParser(description="runs inference")
  parser.add_argument("model_dir",
      help="directory containing a trained model (or full path to weights.pt file)")
  parser.add_argument("--output", "-o", default="output.npy",
      help="output file name")
  parser.add_argument("--fast", "-f", action="store_true", default=False,
      help="fast sampling procedure")
  parser.add_argument("--num_samples", "-n", default=1024, type=int,
      help="number of samples to generate")
  parser.add_argument("-p", "--params_path", default="./params.yaml",
      help="path to the file containing model parameters")
  parser.add_argument("--normalized", "-norm", action="store_true", default=False,
      help="perform reverse normalization on the generated samples")
  main(parser.parse_args())
