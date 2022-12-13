import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--neuron', type=str, default='binary')
parser.add_argument('--perturbation', type=int, default=1)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_steps', type=int, default=9600000)
parser.add_argument('--r_idx', type=int, default=10, help='r = [5e-4, 0.0025, 0.0050, 0,0075, \
                                                           0.0100, ..., 0.0350] --> 12 strengths')

args = parser.parse_args(args=[])
#args = parser.parse_args()

neuron_type = args.neuron
perturbation = args.perturbation
n_neurons = args.n_neurons
n_steps = args.n_steps
r_idx = args.r_idx

rs = np.load('data/recurrent_strength.npy')
thresholds = np.load('data/threshold_{}.npy'.format(neuron_type))

# r = [5e-4, 0.0025, 0.0050, 0,0075, 0.0100, ..., 0.0350] > 12 strengths
r, threshold = rs[r_idx], thresholds[r_idx]

sig1, sig2 = 6.98, 7
a1, a2 = 1, 1.0005
tau, dt = 0.01, 0.0001

# Create weight profile
W = np.zeros((n_neurons, n_neurons))
for i in range(n_neurons):
    for j in range(n_neurons):
        distance = min(np.abs(i-j), n_neurons - np.abs(i-j))
        W[i,j] = a1 * (np.exp(-(distance)**2 / (2*(sig1**2))) - a2 * np.exp(-(distance)**2 / (2*(sig2**2))))
# Noise
b = 0.001
noise_sd, noise_sparsity = 0.3, 1.5
alpha = 0.001
# Initial activations
s = np.zeros(n_neurons) 

def spike_plot(spikes, binning, aspect, suffix):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    num_neurons = spikes.shape[0]
    time = spikes.shape[1] / 1e4
    end_step = int((spikes.shape[1] // binning) * binning)
    spikes = spikes[:, :end_step].reshape(num_neurons, -1, binning)
    spikes_binned = spikes.sum(axis=-1).reshape(num_neurons, -1)

    extent = [0, time, 0, num_neurons]
    axes.set_aspect(aspect)

    im = axes.imshow(spikes_binned, cmap='binary', extent=extent)
    axes.tick_params(axis='x', labelsize=10)
    axes.tick_params(axis='y', labelsize=10)

    cax = fig.add_axes([axes.get_position().x1+0.01,axes.get_position().y0,0.02,axes.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(axis='y', labelsize=20)
    
    if not os.path.exists('./fig/raw'):
        os.makedirs('./fig/raw')

    plt.savefig('./fig/raw/spk_{}'.format(suffix), bbox_inches='tight')
    plt.show()
    plt.close('all')

if neuron_type == 'binary':
    spikes = np.empty((n_neurons, n_steps))
    activations = np.empty((n_neurons, n_steps))

    for i in tqdm(range(n_steps)):
        if perturbation:
            if (i % 400) < 200:
                b = 0.001
            else:
                b = 0

        I = (r*np.matmul(s, W) + (b * (1 + noise_sd*np.random.randn(n_neurons)
                                      * (np.random.randn(n_neurons)>noise_sparsity))))
        spike = (I > threshold).astype(int)
        s = s + spike - s/tau*dt

        spikes[:, i] = spike
        activations[:, i] = s
    
    if perturbation:
        spikes = spikes.reshape((n_neurons, -1, 2, 200))
        spikes = spikes[:,:,0,:].reshape(n_neurons, -1)
        activations = activations.reshape((n_neurons, -1, 2, 200))
        activations = activations[:,:,0,:].reshape(n_neurons, -1)
        spike_plot(spikes, 100, 1, 'binary_raw_pt')

    else:
        spike_plot(spikes, 100, 1, 'binary_raw_eq')

elif neuron_type == 'lnp':
    spikes = np.empty((n_neurons, n_steps))
    lambdas = np.empty((n_neurons, n_steps))

    for i in tqdm(range(n_steps)):
        if perturbation:
            if (i % 400) < 200:
                b = 0.001
            else:
                b = 0
        
        I = r*np.matmul(s, W) + b 
        spike = np.random.poisson(32*np.maximum(I, 0))
        s = s + spike - s/tau*dt

        spikes[:, i] = spike
        lambdas[:, i] = 32*np.maximum(I, 0)

    if perturbation:
        spikes = spikes.reshape((n_neurons, -1, 2, 200))
        spikes = spikes[:,:,0,:].reshape(n_neurons, -1)
        lambdas = lambdas.reshape((n_neurons, -1, 2, 200))
        lambdas = lambdas[:,:,0,:].reshape(n_neurons, -1)
        spike_plot(spikes, 100, 1, 'lnp_raw_pt')

    else:
        spike_plot(spikes, 100, 1, 'lnp_raw_eq')

else:
    assert False, 'Invalid neuron type, try "binary or lnp'