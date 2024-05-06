from lagrangian_datatools import ParticleDataset
import numpy as np

# Loading the data
filename = "../../data/velocities.npy"
d1 = ParticleDataset(npy_filename=filename, root_dir=".")
random_seed = 42
np.random.seed(random_seed)
N = 100000

# Subsampling from a numpy array
# create array of random indices
random_indices = np.random.randint(0, N,size=N)
# subsample
subsampled_data = d1.data[random_indices]
print(subsampled_data.shape)
# save as a npy file
# replace with custom path and filename
np.save("../../data/subsampled_velocities.npy", subsampled_data)
