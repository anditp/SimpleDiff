import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

#%%
v = np.load("/Users/andrei/Desktop/bl/velocities.npy", mmap_mode = "r").transpose((0, 2, 1))[:30000,0,:]
v = np.expand_dims(v, 1)
#%%
w0 = np.load("/Users/andrei/Desktop/bl/samples_30000x1x1800.npz", mmap_mode = "r")["arr_0"]
w0.shape
#%%

with h5py.File('/Users/andrei/Desktop/bl/Lagr_u3c_diffusion_1800.h5', 'r') as h5f:
    u3c = np.array(h5f.get('train'))

w = u3c[:30000,:,0]

#%%
plt.plot(w0[0, 0].flatten())
plt.show()
#%%
class Statistics:
    
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.sample_size = trajectories.shape[0]
        self.T = trajectories.shape[-1]
        print(self.trajectories.shape)
    
    
    def delta_tau(self, tau):
        N = self.T - tau + 1
        x = self.trajectories[:,:,:-tau]
        x_tau = self.trajectories[:,:,tau:]
        
        return (x_tau - x).flatten()
    
    def delta_tau_V(self, tau):
        delta_tau = self.delta_tau(tau)
        delta_tau /= np.std(delta_tau)
        
        
        return delta_tau
    
    
    def Structure(self, p, verbose = False):
        vals = []
        tau = np.logspace(0.0, 3.0, 25, dtype = np.int64)
        for t in tau:
            vals.append(np.mean(self.delta_tau(t) ** p))
        
        if verbose:
            plt.loglog(tau, vals)
        
        return np.array(vals)

#%%
S = Statistics(w)
T = Statistics(w0)

vals_T = T.delta_tau_V(10)
vals_S = S.delta_tau_V(10)

num_bins = 1000

fig, ax = plt.subplots()
ax.hist(vals_T, num_bins, alpha = 0.5, density=True, label = "Sampled")
ax.hist(vals_S, num_bins, alpha = 0.5, density=True, label = "Simulated")
ax.set_xlabel('Value')
ax.set_ylabel('Probability density')
ax.set_title('Histogram')
ax.legend()
plt.yscale("log")

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

#%%

tau = np.logspace(0.0, 3.0, 25, dtype = np.int64)
s4 = S.Structure(4)
s2 = S.Structure(2)
print(len(tau), s2.shape, s4.shape)

s4t = T.Structure(4)
s2t = T.Structure(2)

#%%

fig, ax = plt.subplots()
plt.loglog(tau, s4 / (s2 ** 2), label = "DNS")
plt.loglog(tau, s4t / (s2t ** 2), label = "Simulated")
plt.legend()
plt.show()








