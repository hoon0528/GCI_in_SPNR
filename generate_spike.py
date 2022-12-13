import torch
import numpy as np
import pandas as pd
import os

from scipy.io import loadmat
from torch.utils.data import Dataset
from tqdm import tqdm

root_dir = os.getcwd()
data_dir = root_dir + '/data'

# [sims, tsteps, features, nodes] / x = [nodes, features], edge_index = []
# edge > adjacent matrix [sims, nodes, nodes]
# Using PyTorch Geometric

class data_spike_poisson_rate(Dataset):
    def __init__(self, history, pred_step, dataset, binning, 
                 neuron_type, recording, rm_step, decimated=False):
        '''
        For dataset, use (train, valid or test)
        recording: {eq, pt} Equilibrium or Perturbation
        neuron_type: {LNP, ring}
        '''
        if decimated:
            self.spike_mat = np.load('data/decimation/{}_cprd_{}.npy'.format(neuron_type, rm_step))[:,:4800000]
            self.lam_mat = np.load('data/decimation/{}_cprd_{}.npy'.format(neuron_type, rm_step))[:,:4800000]
        else:
            self.spike_mat = np.load('data/spk_raw_{}_{}.npy'.format(recording, neuron_type))
            self.lam_mat = np.load('data/lam_raw_{}_{}.npy'.format(recording, neuron_type))
        self.pred = pred_step
        self.dataset = dataset
        self.decimated = decimated

        spike_whole = torch.FloatTensor(self.spike_mat)
        lam_whole = torch.FloatTensor(self.lam_mat)
        
        num_neurons = spike_whole.shape[0]

        spike_whole = spike_whole.reshape(num_neurons, -1, binning).sum(dim=-1)
        lam_whole = lam_whole.reshape(num_neurons, -1, binning).sum(dim=-1)

        if binning:
            bin_str = 'bin'
        else:
            bin_str = 'raw'

        if decimated:
            lam_ewm_dir = 'data/lew_dcmd_{}.npy'.format(neuron_type)
        else:
            lam_ewm_dir = 'data/lew_{}_{}_{}.npy'.format(bin_str, recording, neuron_type)

        if not os.path.exists(lam_ewm_dir):
            spk_df = pd.DataFrame(self.spike_mat.T)
            lam_df = spk_df.ewm(alpha=0.01, adjust=False).mean()
            lam_ewm = lam_df.to_numpy().T
            np.save(lam_ewm_dir, lam_ewm)
        else:
            lam_ewm = np.load(lam_ewm_dir)

        total_steps = spike_whole.shape[1]
        window_size = history + pred_step - 1
        
        data_length = total_steps
        idx_tr_end = int(data_length*0.8)
        idx_val_end = int(data_length*0.9)
        idx_test_end = data_length
    
        if dataset == 'train':
            iter_step = int((data_length*0.8 - history) // pred_step)
            spike_whole = spike_whole[:, :idx_tr_end]
            lam_whole = lam_whole[:, :idx_tr_end]
            lam_ewm = lam_ewm[:, :idx_tr_end]
        elif dataset == 'valid':
            iter_step = int((data_length*0.1 - history) // pred_step)
            spike_whole = spike_whole[:, idx_tr_end:idx_val_end]
            lam_whole = lam_whole[:, idx_tr_end:idx_val_end]
            lam_ewm = lam_ewm[:, idx_tr_end:idx_val_end]
        elif dataset == 'test':
            iter_step = int((data_length*0.1 - history) // pred_step)
            spike_whole = spike_whole[:, idx_val_end:idx_test_end]
            lam_whole = lam_whole[:, idx_val_end:idx_test_end]
            lam_ewm = lam_ewm[:, idx_val_end:idx_test_end]
        else:
            assert False, 'Invalid dataset, put train/valid/test'
        
        edge_fullyconnected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        edge_idx_fullyconnected = np.where(edge_fullyconnected)
        edge_idx_fullyconnected = np.array([edge_idx_fullyconnected[0], edge_idx_fullyconnected[1]], dtype=np.int64)

        spike_whole = torch.FloatTensor(spike_whole)
        lam_whole = torch.FloatTensor(lam_whole)
        self.edge_idx = torch.LongTensor(edge_idx_fullyconnected)

        spike_window = []
        spike_target = []
        lam_target = []
        lam_ewm_in = []

        for i in tqdm(range(iter_step)):
            if (dataset == 'train') or (dataset == 'valid'):
                #step = i * history
                step = i * pred_step
                spike_window.append(spike_whole[:, step:step+window_size])
                spike_target.append(spike_whole[:, step+history:step+history+pred_step])
                lam_target.append(lam_whole[:, step+history:step+history+pred_step])
                lam_ewm_in.append(lam_whole[:, step+history:step+history+1])
            elif dataset == 'test':
                step = i * pred_step
                spike_window.append(spike_whole[:, step:step+window_size])
                spike_target.append(spike_whole[:, step+history:step+history+pred_step])
                lam_target.append(lam_whole[:, step+history:step+history+pred_step])
                lam_ewm_in.append(lam_whole[:, step+history:step+history+1])
            else:
                assert False, 'Invalid dataset, put train/valid/test'
        
        spike_window_whole = torch.stack(spike_window, dim=1)
        spike_target_whole = torch.stack(spike_target, dim=1)
        lam_target_whole = torch.stack(lam_target, dim=1)
        lam_ewm_in = torch.stack(lam_ewm_in, dim=1)

        self.x = spike_window_whole
        self.y = spike_target_whole
        self.y_lam = lam_target_whole
        self.x_lam = lam_ewm_in
        self.len = spike_window_whole.shape[1]

    def __len__(self):
        return self.len

    def get_adjmat(self):
        return self.edge_idx

    def __getitem__(self, idx):
        if self.dataset == 'train' or self.dataset == 'valid':
            x = self.x[:, idx, :]
            y = self.y[:, idx, :]
            x_lam = self.x_lam[:, idx, :]
            return x, y
        else:
            x = self.x[:, idx, :]
            y = self.y[:, idx, :]
            x_lam = self.x_lam[:, idx, :]
            lam_tar = self.y_lam[:, idx, :]
            return x, y

class mouse_head_direction(Dataset):
    def __init__(self, history, pred_step, dataset):
        '''
        For dataset, use (train, valid or test)
        Session, Mouse Index
        '''
        data_all = loadmat('data/dataHD.mat')
        pref_HD = data_all['prefHD'].reshape(-1)
        hd_idx = np.argsort(pref_HD)
        ang_run = data_all['angRun'][:, 1]
        
        self.spike_train = data_all['Qrun'][:, hd_idx] #[time steps, neurons]
        self.pred = pred_step
        self.pref_HD = torch.FloatTensor(pref_HD[hd_idx])

        spike_whole = torch.FloatTensor(self.spike_train)
        ang_run = torch.FloatTensor(ang_run)

        total_steps, num_neurons = spike_whole.shape[0], spike_whole.shape[1]
        spike_whole = spike_whole.T
        window_size = history + pred_step - 1
        
        data_length = total_steps # For binning purpose
        idx_tr_end = int(data_length*0.8)
        idx_val_end = int(data_length*0.9)
        idx_test_end = data_length

        gts_featmat = spike_whole[:, :idx_tr_end]
        if not os.path.exists('data/mouse'):
            os.makedirs('data/mouse')
        np.save('data/mouse/gts_featmat.npy', gts_featmat.numpy())
    
        if dataset == 'train':
            iter_step = int((data_length*0.8-history) // pred_step)
            spike_whole = spike_whole[:, :idx_tr_end]
            ang_run = ang_run[:idx_tr_end]
        elif dataset == 'valid':
            iter_step = int((data_length*0.1-history) // pred_step)
            spike_whole = spike_whole[:, idx_tr_end:idx_val_end]
            ang_run = ang_run[:idx_tr_end]
        elif dataset == 'test':
            iter_step = int((data_length*0.1-history) // pred_step)
            spike_whole = spike_whole[:, idx_val_end:idx_test_end]
            ang_run = ang_run[:idx_tr_end]
        else:
            assert False, 'Invalid dataset, put train/valid/test'

        # Memory Issue of getting 48K steps during Training Phase 2
        edge_fullyconnected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        edge_idx_fullyconnected = np.where(edge_fullyconnected)
        edge_idx_fullyconnected = np.array([edge_idx_fullyconnected[0], edge_idx_fullyconnected[1]], dtype=np.int64)

        spike_whole = torch.FloatTensor(spike_whole)
        self.edge_idx = torch.LongTensor(edge_idx_fullyconnected)

        spike_window = []
        spike_target = []
        ang_window = []

        for i in tqdm(range(iter_step)):
            if (dataset == 'train') or (dataset == 'valid'):
                step = i * pred_step
                spike_window.append(spike_whole[:, step:step+window_size])
                spike_target.append(spike_whole[:, step+history:step+history+pred_step])
                ang_window.append(ang_run[step:step+window_size])
            elif dataset == 'test':
                step = i * pred_step
                spike_window.append(spike_whole[:, step:step+window_size])
                spike_target.append(spike_whole[:, step+history:step+history+pred_step])
                ang_window.append(ang_run[step:step+window_size])
            else:
                assert False, 'Invalid dataset, put train/valid/test'
        
        spike_window_whole = torch.stack(spike_window, dim=1)
        spike_target_whole = torch.stack(spike_target, dim=1)
        ang_window_whole = torch.stack(ang_window, dim=0)

        # [num_neurons, num_data, time_steps]
        self.x = spike_window_whole
        self.y = spike_target_whole
        self.ang = ang_window_whole
        self.len = spike_window_whole.shape[1]

    def __len__(self):
        return self.len

    def get_adjmat(self):
        return self.edge_idx
    
    def get_pref_HD(self):
        return self.pref_HD

    def __getitem__(self, idx):
        x = self.x[:, idx, :]
        y = self.y[:, idx, :]
        ang = self.ang[idx, :]
        return x, y, ang

class random_configurations(Dataset):
    def __init__(self, p, num_neurons, history, num_batches, batch_size):
        '''
            Generate random spike with (1-p, p) ~ {0, 1}
        '''
        total_steps = history*num_batches*batch_size
        spk_np = np.random.choice(2, size=(num_neurons, total_steps), p=(1-p, p))
        spk_df = pd.DataFrame(spk_np.T)
        lam_df = spk_df.ewm(alpha=0.01, adjust=False).mean()
        lam_ewm = lam_df.to_numpy().T

        spk_np = spk_np.reshape(num_neurons, num_batches, batch_size, history)
        lam_ewm = lam_ewm.reshape(num_neurons, num_batches, batch_size, history)
        lam_ewm_in = lam_ewm[:, :, :, history-1:history]

        spike_whole = torch.FloatTensor(spk_np)
        lam_whole = torch.FloatTensor(lam_ewm)
        lam_in = torch.FloatTensor(lam_ewm_in)
                    
        edge_fullyconnected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        edge_idx_fullyconnected = np.where(edge_fullyconnected)
        edge_idx_fullyconnected = np.array([edge_idx_fullyconnected[0], edge_idx_fullyconnected[1]], dtype=np.int64)
        self.edge_idx = torch.LongTensor(edge_idx_fullyconnected)
    
        self.x = spike_whole
        self.y_lam = lam_whole
        self.x_lam = lam_in
        self.len = num_batches

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[:, idx, :, :]
        y_lam = self.y_lam[:, idx, :, :]
        x_lam = self.x_lam[:, idx, :, :]
        return x, y_lam, x_lam, self.edge_idx