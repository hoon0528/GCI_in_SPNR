import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils_plot import *
from baselines.seqnmf import seqnmf
from baselines.tensortools import tensortools as tca

#Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gen_model', type=str, default='binary')
parser.add_argument('--mode', type=str, default='plot')
args = parser.parse_args(args=[])

#Hyperparameters
n_neurons = 100
n_trials = 10
n_factors = 100
seq_length = 2
seq_lambda = 1e-3

#Generate Folders
save_folder = 'baselines/result/{}'.format(args.gen_model)
model_folder = ['model/seqNMF', 'model/TCA']
gen_model = args.gen_model

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for dirs in model_folder:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

if args.mode == 'experiment':
    #load&preprocess neural spike data
    data_seqNMF = np.load('data/spk_raw_eq_{}.npy'.format(gen_model))[:, :3840000]
    data_tca = pd.DataFrame(data_seqNMF.T)
    data_tca = data_tca.ewm(alpha=0.01, adjust=False).mean()
    data_tca = data_tca.to_numpy().T.reshape(n_neurons, n_trials, -1) #N, K, T

    #TCA (fitting)
    ensemble = tca.Ensemble(fit_method='ncp_hals')
    ensemble.fit(data_tca, ranks=n_factors, replicates=1)

    tca_factors_list = ensemble.factors(rank=n_factors)
    tca_factors = tca_factors_list[0]

    #seqNMF (fitting)
    seq_neuron_factors, seq_time_factors, error_log, loadings, recon_score_sqenmf = \
        seqnmf.seqnmf(data_seqNMF, n_factors, seq_length, seq_lambda) 

    #TCA Results
    recon = tca_factors.full()

    tca_neuron_factors = tca_factors.__getitem__(0)
    tca_trial_factors = tca_factors.__getitem__(1)
    tca_time_factors = tca_factors.__getitem__(2)

    recon_error_tca = np.mean((data_tca - recon)**2)
    recon_score_tca = 1 - recon_error_tca/((data_tca**2).mean())

    #Save results
    np.save('baselines/result/{}/seqNMF_N.npy'.format(gen_model), seq_neuron_factors)
    np.save('baselines/result/{}/seqNMF_T.npy'.format(gen_model), seq_time_factors)
    np.save('baselines/result/{}/tca_N.npy'.format(gen_model), tca_neuron_factors)
    np.save('baselines/result/{}/tca_K.npy'.format(gen_model), tca_trial_factors)
    np.save('baselines/result/{}/tca_T.npy'.format(gen_model), tca_time_factors)

    file = 'baselines/result/{}/log.txt'.format(gen_model)

    if os.path.isfile(file):
        exp_log = open(file, 'a')
    else:
        exp_log = open(file, 'w')

    err_log = 'Reconstruction Rate - seqNMF:{:5f}, TCA:{:5f}'.format(recon_score_sqenmf, recon_score_tca)
    exp_log.write(err_log)
    exp_log.close()
    print(err_log)

########## for plotting purpose  ###############

elif args.mode == 'plot':
    seq_N = np.load('baselines/result/{}/seqNMF_N.npy'.format(gen_model))
    seq_T = np.load('baselines/result/{}/seqNMF_T.npy'.format(gen_model))
    tca_N = np.load('baselines/result/{}/tca_N.npy'.format(gen_model))
    tca_K = np.load('baselines/result/{}/tca_K.npy'.format(gen_model))
    tca_T = np.load('baselines/result/{}/tca_T.npy'.format(gen_model))

    tca_edges, seq_edges = [], []

    for i in tqdm(range(n_factors)):
        tca_edge_feature = tca_N[:, i]
        tca_edge = np.outer(tca_edge_feature, tca_edge_feature)
        
        seq_edge_feature1 = seq_N[:, i, 0]
        seq_edge_feature2 = seq_N[:, i, 1]
        seq_edge = np.outer(seq_edge_feature1, seq_edge_feature2)
        
        tca_edges.append(tca_edge)
        seq_edges.append(seq_edge)

    tca_W_list, seq_W_list = [], []

    for i in range(n_factors):
        tca_W = tca_T[:, i].sum() * tca_K[:, i].sum() * tca_edges[i]
        tca_W_list.append(tca_W)

        seq_W = seq_T[i, :].sum() * seq_edges[i]
        seq_W_list.append(seq_W)

    tca_z = np.array(tca_W_list).mean(axis=0)
    seq_z = np.array(seq_W_list).mean(axis=0)

    tca_z = tca_z[~np.eye(100, dtype=bool)].reshape(-1, 1)
    seq_z = seq_z[~np.eye(100, dtype=bool)].reshape(-1, 1)

    np.save('model/TCA/z_{}.npy'.format(gen_model), tca_z)
    np.save('model/seqNMF/z_{}.npy'.format(gen_model), seq_z)

else:
    assert False, 'Invalid mode try {experiment or plot}'

#21mlp, 22conv > binary, 20pred
#23mlp, 24conv > lnp, 20pred
#17mlp, 18conv > binary, recons
#19mlp, 20conv > lnp, recons