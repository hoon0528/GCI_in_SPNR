import os
import torch
import numpy as np

from tqdm import tqdm
from generate_spike import *
from scipy.optimize import minimize
from model.nri_vsc_wo_pinputs import RNN_DEC_lam, RNN_DEC_log, GNN_ENC_mousehd
from torch_geometric import utils as pyg_utils
from torch.utils.data import Dataset, DataLoader
from model.gts_adj import gts_adj_inf

import matplotlib.pyplot as plt

def make_z_sym(z, num_nodes, batch_num):
    z = z.reshape(batch_num, num_nodes, num_nodes-1)
    for i in range(num_nodes-1):
        z[:, i+1:num_nodes, i] = z[:, i, i:num_nodes-1]
    z = z.reshape(-1, 1)
    return z

def make_z_sym_nri(z, num_nodes, batch_num, edge_types):
    z = z.reshape(batch_num, num_nodes, num_nodes-1, edge_types)
    for i in range(num_nodes-1):
        z[:, i+1:num_nodes, i, :] = z[:, i, i:num_nodes-1, :]
    z = z.reshape(-1, edge_types)
    return z

def make_z_sym_gts(z, edge_idx, num_nodes, mode):
    if mode == 'transpose':
        z = z.reshape(-1)
        adj = pyg_utils.to_dense_adj(edge_idx, edge_attr=z)
        adj = adj.squeeze()
        adj = (adj + adj.t()) / 2 
        edge_idx, z = pyg_utils.dense_to_sparse(adj)
    elif mode == 'ltmatonly':
        z = z.reshape(num_nodes, num_nodes-1)
        for i in range(num_nodes-1):
            z[i+1:num_nodes, i] = z[i, i:num_nodes-1]
        z = z.reshape(-1)
    else:
        assert False, 'Invalid Mode'
    return edge_idx, z

def calc_l1_dev_scaler(x, w_hat, w):
    l1_dev = np.abs(x*w_hat - w - np.mean(x*w_hat - w))
    return np.mean(l1_dev)

def calc_l1_dev_bias(x, w_hat, w):
    l1_dev = np.square(w_hat + x - w)
    return np.mean(l1_dev)

def calc_l1_dev_scale_factor_l2_loss(w_hat, w_tar):
    num_nodes = w_hat.shape[0]
    w_hat_ori, w_tar_ori = np.empty((num_nodes, num_nodes)), np.empty((num_nodes, num_nodes))

    for i in range(num_nodes):
        w_hat_ori[i] = np.roll(w_hat[i], -i)
        w_tar_ori[i] = np.roll(w_tar[i], -i)

    w_hat_vec = np.mean(w_hat_ori, axis=0)
    w_tar_vec = np.mean(w_tar_ori, axis=0)
    w_hat_vec, w_tar_vec = w_hat_vec.reshape(-1), w_tar_vec.reshape(-1)
 
    k_argmin = None
    l1_dev_min = np.inf

    for i in range(num_nodes):        
        if (w_hat_vec[i] - np.mean(w_hat_vec)) != 0:
            k = (w_tar_vec[i] + np.mean(w_tar_vec)) / (w_hat_vec[i] - np.mean(w_hat_vec))
            l1_dev = np.linalg.norm(w_hat_vec*k - w_tar_vec, ord=1)
            if l1_dev < l1_dev_min:
                l1_dev_min = l1_dev
                k_argmin = k
    
    l2_loss = np.linalg.norm(w_hat_vec*k_argmin - w_tar_vec, ord=2)
    l2_tar = np.linalg.norm(w_tar_vec, ord=2)
    return k_argmin, l2_loss/l2_tar, w_hat_vec, w_tar_vec

def scale_w_hat_scipy(w_hat, w_tar):
    num_nodes = w_hat.shape[0]
    w_hat_ori, w_tar_ori = np.empty((num_nodes, num_nodes)), np.empty((num_nodes, num_nodes))

    for i in range(num_nodes):
        w_hat_ori[i] = np.roll(w_hat[i], -i)
        w_tar_ori[i] = np.roll(w_tar[i], -i)

    w_hat_vec = np.mean(w_hat_ori, axis=0)
    w_tar_vec = np.mean(w_tar_ori, axis=0)
    w_hat_vec, w_tar_vec = w_hat_vec.reshape(-1), w_tar_vec.reshape(-1)
    
    k_init = np.concatenate((np.random.rand(1), np.zeros(1)))
    k_scaler = minimize(calc_l1_dev_scaler, k_init[0], args=(w_hat_vec, w_tar_vec)).x
    k_bias = minimize(calc_l1_dev_bias, k_init[1], args=(w_hat_vec*k_scaler, w_tar_vec)).x
    k_argmin = np.concatenate((k_scaler, k_bias))
    
    w_hat_norm = w_hat_ori * k_scaler + k_bias
    w_hat_std = np.std(w_hat_norm, axis=0)

    l2_loss = np.linalg.norm(w_hat_vec*k_scaler + k_bias - w_tar_vec, ord=2)
    l2_tar = np.linalg.norm(w_tar_vec, ord=2)
    return k_argmin, l2_loss/l2_tar, w_hat_vec, w_tar_vec, w_hat_std

def spec_decompose(mat, count):
    mat_spec = np.zeros_like(mat)
    
    mat_normed = (np.abs(mat) - np.abs(mat).min()) / (np.abs(mat).max() - np.abs(mat).min())
    lam, vec_all = np.linalg.eigh(mat_normed)
    arg_descending = np.abs(lam).argsort()[::-1]

    lam_sorted = lam[arg_descending]
    vec_all_sorted = vec_all[:, arg_descending]

    for i in range(count):
        idx = arg_descending[i]
        vec = vec_all[:, idx:idx+1]
        mat_spec = mat_spec + lam[idx]*np.matmul(vec, vec.T)
    
    return lam_sorted, vec_all_sorted, mat_spec

def get_edgeidx_by_batchsize(edge, batch_size, num_nodes, device):
    size = [batch_size] + list(edge.shape)
    edge_expand = torch.empty(size)
    edge_expand = edge_expand.to(device)
    
    for i in range(batch_size):
        edge_expand[i, :, :] = edge + num_nodes*i
    
    edge_expand = edge_expand.permute(1, 0, 2)
    edge_expand = edge_expand.reshape(edge.shape[0], -1)
    return edge_expand.long()

def z_to_adj(z, nodes, device):
    adj = torch.zeros(nodes * nodes)
    
    for i in range(nodes - 1):
        adj[i*(nodes+1) + 1:(i+1)*(nodes+1)] = z[i*nodes: (i+1)*nodes]
    
    adj = adj.reshape(nodes, nodes)
    adj = adj.to(device)
    return adj.long()

def adj_to_edge_idx_pyg(adj):
    adj = torch.where(adj)
    adj = torch.stack((adj[0], adj[1]), dim=0)
    return adj

def z_to_adj_plot(z, nodes):
    adj = np.zeros(nodes * nodes)
    
    for i in range(nodes - 1):
        adj[i*(nodes+1) + 1:(i+1)*(nodes+1)] = z[i*nodes: (i+1)*nodes] #dim 98 why?
    
    adj = adj.reshape(nodes, nodes)
    return adj

def to_dec_batch(spike_whole, nodes, history, pred_step, p2_bs):
    total_steps = spike_whole.shape[1]
    window_size = history + pred_step - 1
    iter_step = int((total_steps - history) / pred_step)

    spike_window = []
    spike_target = []

    for i in range(iter_step):
        step = i * pred_step
        spike_window.append(spike_whole[:, step:step+window_size])
        spike_target.append(spike_whole[:, step+history:step+history+pred_step])
    
    spike_window_whole = torch.stack(spike_window, dim=1) #[100, iter_step, 219]
    spike_target_whole = torch.stack(spike_target, dim=1) #[100, iter_step, 20]

    whole_cut = int(spike_window_whole.shape[1] / p2_bs) * p2_bs
    
    spike_window_whole = spike_window_whole[:, :whole_cut, :]
    spike_target_whole = spike_target_whole[:, :whole_cut, :]     
    spike_window_whole = spike_window_whole.reshape(nodes, -1, p2_bs, window_size) #[100, iters, bs, window]
    spike_target_whole = spike_target_whole.reshape(nodes, -1, p2_bs, pred_step)

    return spike_window_whole, spike_target_whole

def to_dec_batch_ptar(spike_whole, nodes, history, pred_step, p2_bs):
    total_steps = spike_whole.shape[1]
    window_size = history + pred_step - 1
    iter_step = int(total_steps / history)

    spike_window = []
    spike_target = []

    for i in range(iter_step):
        step = i * history
        spike_window.append(spike_whole[:, step:step+window_size])
        spike_target.append(spike_whole[:, step+history:step+history+pred_step])
    
    spike_window_whole = torch.stack(spike_window, dim=1) #[100, iter_step, 219]
    spike_target_whole = torch.stack(spike_target, dim=1) #[100, iter_step, 20]

    whole_cut = int(spike_window_whole.shape[1] / p2_bs) * p2_bs
    
    spike_window_whole = spike_window_whole[:, :whole_cut, :]
    spike_target_whole = spike_target_whole[:, :whole_cut, :]     
    spike_window_whole = spike_window_whole.reshape(nodes, -1, p2_bs, window_size) #[100, iters, bs, window]
    spike_target_whole = spike_target_whole.reshape(nodes, -1, p2_bs, pred_step)

    return spike_window_whole, spike_target_whole

def to_dec_batch_ptar_plot(spike_whole, nodes, history, pred_step, p2_bs):
    total_steps = spike_whole.shape[1]
    window_size = history + pred_step - 1
    iter_step = int((total_steps - history) / pred_step)

    spike_window = []
    spike_target = []

    for i in range(iter_step):
        step = i * pred_step
        spike_window.append(spike_whole[:, step:step+window_size])
        spike_target.append(spike_whole[:, step+history:step+history+pred_step])
    
    spike_window_whole = torch.stack(spike_window, dim=1) #[100, iter_step, 219]
    spike_target_whole = torch.stack(spike_target, dim=1) #[100, iter_step, 20]

    whole_cut = int(spike_window_whole.shape[1] / p2_bs) * p2_bs
    
    spike_window_whole = spike_window_whole[:, :whole_cut, :]
    spike_target_whole = spike_target_whole[:, :whole_cut, :]     
    spike_window_whole = spike_window_whole.reshape(nodes, -1, p2_bs, window_size) #[100, iters, bs, window]
    spike_target_whole = spike_target_whole.reshape(nodes, -1, p2_bs, pred_step)

    return spike_window_whole, spike_target_whole

def decimation_method_iterX(inputs, is_spike, is_manual, rm_step=0):
    if is_spike:
        lam_whole = inputs.clone()
        spike_whole = torch.poisson(lam_whole)
        pre_rmv = 2*spike_whole - 1
    else:
        pre_rmv = inputs.clone()

    num_neurons = pre_rmv.shape[0]
    total_steps = pre_rmv.shape[1]
    # Get spike index
    nonzero_idx = torch.where(pre_rmv.any(dim=0))[0]
    nonzero_idx = nonzero_idx.detach().cpu().numpy()
    delta_R_spike = []
    # Get zeros index
    zeros_idx = torch.where(~pre_rmv.any(dim=0))[0]
    rand_perm = torch.randperm(torch.numel(zeros_idx))
    zeros_idx = zeros_idx[rand_perm]

    corr_org = torch.matmul(pre_rmv, pre_rmv.transpose(1, 0)) / num_neurons
    
    for idx_s in tqdm(nonzero_idx):
        s = pre_rmv[:, idx_s].reshape(-1, 1)
        delta_R = decimation_removal(corr_org, s, total_steps, False)
        delta_R_spike.append(delta_R.detach().cpu().item())

    delta_R_spike = np.array(delta_R_spike)
    idx_spk_sort = np.argsort(delta_R_spike)[::-1]
    nonzero_idx = nonzero_idx[idx_spk_sort]
    delta_R_spike = delta_R_spike[idx_spk_sort]

    delta_R_return = delta_R_spike.copy()
    
    idx_config_removal = []
    R_ipr = []
    lam_list = []

    C_crnt = corr_org
    m = total_steps

    if is_manual:
        for _ in tqdm(range(rm_step)):
            if torch.numel(zeros_idx) == 0:
                delta_R_zeros = -np.inf    
            else:
                s_z = pre_rmv[:, zeros_idx[0]].reshape(-1, 1)
                delta_R_zeros = decimation_removal(C_crnt, s_z, m, False)
            
            if delta_R_spike.max() > delta_R_zeros:
                idx_config_removal.append(nonzero_idx[0])
                s_sm = pre_rmv[:, nonzero_idx[0]].reshape(-1, 1)
                C = decimation_removal(C_crnt, s_sm, m, True)
                nonzero_idx = nonzero_idx[1:]
                delta_R_spike = delta_R_spike[1:]
            else:
                zeros_rmv = zeros_idx[0]
                idx_config_removal.append(zeros_rmv)
                zeros_idx = zeros_idx[1:]
                s_zm = pre_rmv[:, zeros_rmv].reshape(-1, 1)
                C = decimation_removal(C_crnt, s_zm, m, True)

            R_crnt, lam = get_R_ipr(C_crnt)
            R_next, _ = get_R_ipr(C)
            R_ipr.append(R_crnt.detach().cpu().numpy())
            lam_list.append(lam.detach().cpu().numpy())
            C_crnt = C
            m = m - 1

        idx_remaining = np.ones(total_steps)
        idx_remaining[idx_config_removal] = 0
        idx_remaining = np.where(idx_remaining == 1)[0]
        
        rmvd = pre_rmv[:, idx_remaining]
        rmvd = rmvd.detach().cpu().numpy()
        lam_np = np.stack(lam_list, axis=0)
        R_ipr = np.stack(R_ipr, axis=0)

        return rmvd, lam_np, R_ipr, delta_R_return, nonzero_idx
        
    else:
        pbar = tqdm(total=total_steps)
        while(True):
            if torch.numel(zeros_idx) == 0:
                delta_R_zeros = -np.inf    
            else:
                s_z = pre_rmv[:, zeros_idx[0]].reshape(-1, 1)
                delta_R_zeros = decimation_removal(C_crnt, s_z, m, False)
            
            if delta_R_spike.max() > delta_R_zeros:
                idx_config_removal.append(nonzero_idx[0])
                s_sm = pre_rmv[:, nonzero_idx[0]].reshape(-1, 1)
                C = decimation_removal(C_crnt, s_sm, m, True)
                nonzero_idx = nonzero_idx[1:]
                delta_R_spike = delta_R_spike[1:]
            else:
                zeros_rmv = zeros_idx[0]
                idx_config_removal.append(zeros_rmv)
                zeros_idx = zeros_idx[1:]
                s_zm = pre_rmv[:, zeros_rmv].reshape(-1, 1)
                C = decimation_removal(C_crnt, s_zm, m, True)

            R_crnt, lam = get_R_ipr(C_crnt)
            R_next, _ = get_R_ipr(C)
            R_ipr.append(R_crnt.detach().cpu().numpy())
            lam_list.append(lam.detach().cpu().numpy())
            C_crnt = C
            m = m - 1

            pbar.update(1)
            if (R_next - R_crnt) < 0:
                idx_remaining = np.ones(total_steps)
                idx_remaining[idx_config_removal] = 0
                idx_remaining = np.where(idx_remaining == 1)[0]
                rmvd = pre_rmv[:, idx_remaining]
                rmvd = rmvd.detach().cpu().numpy()

                lam_np = np.stack(lam_list, axis=0)
                R_ipr = np.stack(R_ipr, axis=0)
                pbar.close()
                return rmvd, lam_np, R_ipr, delta_R_return, nonzero_idx

def decimation_method_iterO(spikes):
    spike_whole = spikes.clone()
    num_neurons = spike_whole.shape[0]
    total_steps = spike_whole.shape[1]

    corr_org = torch.matmul(spikes, spikes.transpose(1, 0)) / num_neurons
    
    spike_idx = torch.where(spike_whole.any(axis=0))[0]
    zeros_idx = torch.where(~spike_whole.any(axis=0))[0]
    rand_perm = torch.randperm(torch.numel(zeros_idx))
    zeros_idx = zeros_idx[rand_perm]
    
    idx_config_removal = []
    R_ipr = []
    lam_list = []
    
    C_crnt = corr_org
    m = total_steps
    while(True):
        delta_R_spike = []
    
        for idx_s in tqdm(spike_idx):
            s = spike_whole[:, idx_s].reshape(-1, 1)
            delta_R = decimation_removal(C_crnt, s, m, False)
            delta_R_spike.append(delta_R.detach().cpu().item())
    
        if torch.numel(zeros_idx) == 0:
            delta_R_zeros = -np.inf    
        else:
            s_z = spike_whole[:, zeros_idx[0]].reshape(-1, 1)
            delta_R_zeros = decimation_removal(C_crnt, s_z, m, False)
    
        delta_R_spike = np.array(delta_R_spike)
        if delta_R_spike.max() > delta_R_zeros:
            print(delta_R_spike.max())
            idx_spk_max = delta_R_spike.argmax()
            idx_spk_rmv = spike_idx[idx_spk_max]
            spike_idx = spike_idx[spike_idx != idx_spk_rmv]
            idx_config_removal.append(idx_spk_rmv)
            s_sm = spike_whole[:, idx_spk_rmv].reshape(-1, 1)
            C = decimation_removal(C_crnt, s_sm, m, True)
    
        else:
            print(delta_R_zeros)
            zeros_rmv = zeros_idx[0]
            idx_config_removal.append(zeros_rmv)
            zeros_idx = zeros_idx[1:]
            s_zm = spike_whole[:, zeros_rmv].reshape(-1, 1)
            C = decimation_removal(C_crnt, s_zm, m, True)
    
        R_crnt, lam = get_R_ipr(C_crnt)
        R_next, _ = get_R_ipr(C)
        R_ipr.append(R_crnt.detach().cpu().numpy())
        lam_list.append(lam.detach().cpu().numpy())
    
        C_crnt = C
        m = m - 1
    
        if (R_next - R_crnt) < 0:
            idx_remaining = np.ones(total_steps)
            idx_remaining[idx_config_removal] = 0
            idx_remaining = np.where(idx_remaining == 1)[0]
            spike_whole = spike_whole[:, idx_remaining]
            lam_np = np.stack(lam_list, axis=0)
            R_ipr = np.stack(R_ipr, axis=0)
    
            return spike_whole, lam_np, R_ipr

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
    plt.savefig('./fig/decimation/spk_{}'.format(suffix), bbox_inches='tight')
    plt.show()
    plt.close('all')

def save_decimation(spikes, rm_step, p_spars, rm_mode, neuron_type, device, rm_zeros, if_rmv_zeros=False):
    p_str = str(p_spars).replace('.', '')
    if if_rmv_zeros:
        suffix = '{}_cprd_{}'.format(neuron_type, rm_step)
    else:
        suffix = '{}_{}_{}_{}'.format(neuron_type, rm_mode, rm_step, p_str)
    print(suffix)
    if if_rmv_zeros:
        spk_prd, _ = decimation_method_iterX(spikes, rm_zeros, rm_mode, p_spars, device, if_rmv_zeros)
    else:
        spk_prd, _ = decimation_method_iterX(spikes, rm_step, rm_mode, p_spars, device)
    spike_plot(spk_prd, 100, 1, suffix)
    np.save('data/decimation/{}.npy'.format(suffix), spk_prd)

def check_isi(spikes):
    num_spikes = np.count_nonzero(spikes)
    spk_steps = len(np.where(spikes.any(axis=0))[0])
    total_steps = spikes.shape[1]
    isi = num_spikes / spikes.size
    target_isi = 1/160
    num_removal = int((spikes.size - (num_spikes / target_isi)) / 100)

    print('Total steps: {}, number of spike steps is {}'.format(total_steps, spk_steps))
    print('Try to meet ISI condition, {} steps of zero steps should be removed'.format(num_removal))
    return isi, num_removal, spk_steps, total_steps

def remove_zeros(spikes, removal_count):
    spike_whole = spikes.copy()
    total_steps = spike_whole.shape[1]
    
    zeros_idx = np.where(~spike_whole.any(axis=0))[0]
    rmv_idx = np.random.choice(zeros_idx, size=removal_count, replace=False)
    
    idx_remaining = np.ones(total_steps)
    idx_remaining[rmv_idx] = 0
    idx_remaining = np.where(idx_remaining == 1)[0]
    spike_whole = spike_whole[:, idx_remaining]
    
    return spike_whole

def get_R_ipr(C):
    num_neurons = C.shape[0]
    lam, _ = torch.linalg.eigh(C)
    lam_scaler = num_neurons / torch.abs(lam).sum()
    lam = lam * lam_scaler
    sig = torch.sqrt(torch.abs(lam))
    r_i = sig / sig.sum()
    R_ipr = 1 / (r_i ** 2).sum()
    return R_ipr, torch.abs(lam)

def decimation_removal(C, single_config, num_configs, if_update_C):
    lam, vec_all = torch.linalg.eigh(C)
    num_neurons = C.shape[0]
    vec_all = vec_all.transpose(1, 0).reshape(-1, 1, num_neurons)
    vec_all_T = vec_all.permute(0, 2, 1)
    
    delta_C = (C - torch.matmul(single_config, single_config.transpose(1, 0))) / (num_configs - 1)
    if if_update_C:
        C_rmvd = C + delta_C
        return C_rmvd
    else:
        delta_lam = torch.matmul(vec_all, delta_C)
        delta_lam = torch.matmul(delta_lam, vec_all_T)
        delta_R = (1/torch.sqrt(torch.abs(lam)) * delta_lam.squeeze()).sum()
    return delta_R

def plot_lambdas(lam, suffix):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    
    num_rmvs = lam.shape[0]
    num_neurons = lam.shape[1]
    x = np.arange(num_neurons) + 1

    for i in range(num_rmvs):
        axes.plot(x, lam[i, :])

    plt.savefig('./fig/decimation/lam_{}'.format(suffix), bbox_inches='tight')
    plt.close('all')

def plot_lamlist(lam_list, label_list, suffix):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    
    num_neurons = lam_list[0].size
    x = np.arange(num_neurons) + 1

    for i in range(len(lam_list)):
        axes.plot(x, lam_list[i], label=label_list[i])

    plt.legend(loc=2, prop={'size': 20})
    plt.savefig('./fig/decimation/lam_{}'.format(suffix), bbox_inches='tight')
    plt.close('all')

def plot_Ripr(R_ipr, suffix):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    x = np.arange(R_ipr.size) + 1
    axes.plot(x, R_ipr)
    plt.savefig('./fig/decimation/Ripr_{}'.format(suffix), bbox_inches='tight')
    plt.close('all')

def save_directory(args):
    check_neuron_type = (args.neurons == 'lnp') or (args.neurons == 'binary')
    check_decoder = (args.decoder == 'log') or (args.decoder == 'lam')
    assert check_neuron_type, 'Invalid neuron type, try with {lnp or binary,}'
    assert check_decoder, 'Invalid decoder type, try with {log or lam}'

    if args.dataset == 'neural_spike':
        save_folder = 'model/neural_spike/exp{}'.format(args.exp_num)
    elif args.dataset == 'mouse_head_direction':
        save_folder = 'model/mouse_hd/exp{}'.format(args.exp_num)
    else:
        assert False, 'Invaild dataset, try {neural_spike of mouse_hd}'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    directory_list = []
    if args.experiment == 'perturb_joint':
        directory_list.append(save_folder + '/enc_1.pt')
        directory_list.append(save_folder + '/enc_2.pt')
        directory_list.append(save_folder + '/dec_1.pt')
        directory_list.append(save_folder + '/dec_2.pt')
        directory_list.append(save_folder + '/loss.npy')
        directory_list.append(save_folder + '/log.txt')
        directory_list.append(save_folder + '/z_1.npy')
        directory_list.append(save_folder + '/z_2.npy')
    else:
        directory_list.append(save_folder + '/enc.pt')
        directory_list.append(save_folder + '/dec.pt')
        directory_list.append(save_folder + '/loss.npy')
        directory_list.append(save_folder + '/log.txt')
        directory_list.append(save_folder + '/z.npy')
    return save_folder, directory_list

def plot_directory(args):
    with_hd_input = args.dataset == 'mouse_head_direiction' and args.experiment == 'with_hd_input'

    if args.dataset == 'neural_spike':
        plot_folder = 'plot/neural_spike/exp{}'.format(args.exp_num)
        fig_folder = 'fig/neural_spike/exp{}'.format(args.exp_num)
    elif args.dataset == 'mouse_head_direction':
        plot_folder = 'plot/mouse_hd/exp{}'.format(args.exp_num)
        fig_folder = 'fig/mouse_hd/exp{}'.format(args.exp_num)
    else:
        assert False, 'Invaild dataset, try {neural_spike of mouse_hd}'

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    directory_list = []
    if args.experiment == 'perturb_joint':
        directory_list.append(plot_folder + '/spk_tar_p1.npy')
        directory_list.append(plot_folder + '/spk_pred_p1.npy')
        directory_list.append(plot_folder + '/lam_tar_p1.npy')
        directory_list.append(plot_folder + '/lam_pred_p1.npy')
        directory_list.append(plot_folder + '/z1.npy')
        directory_list.append(plot_folder + '/spk_tar_p2.npy')
        directory_list.append(plot_folder + '/spk_pred_p2.npy')
        directory_list.append(plot_folder + '/lam_tar_p2.npy')
        directory_list.append(plot_folder + '/lam_pred_p2.npy')
        directory_list.append(plot_folder + '/z2.npy')        
    else:
        directory_list.append(plot_folder + '/spk_tar.npy')
        directory_list.append(plot_folder + '/spk_pred.npy')
        directory_list.append(plot_folder + '/lam_tar.npy')
        directory_list.append(plot_folder + '/lam_pred.npy')
        directory_list.append(plot_folder + '/z.npy')
        
        if with_hd_input:
            directory_list.append(plot_folder + '/ang_pred.npy')
            directory_list.append(plot_folder + '/ang_tar.npy')
    return fig_folder, plot_folder, directory_list

def load_plot_file(plot_directory):
    file_list = []
    for dirs in plot_directory:
        file_list.append(np.load(dirs))
    return file_list

def load_models(num_neurons, args, device):
    tr_length = int(args.gts_totalstep / args.bin)
    adj_inf_model = gts_adj_inf(num_neurons, args.hid_dim, args.out_channel, args.kernal_x_1, args.kernal_x_2,
                                args.stride_x_1, args.stride_x_2, tr_length)
    batch_size = args.phase1_batchsize

    with_hd_input = args.dataset == 'mouse_head_direction' and args.experiment == 'with_hd_input'
    no_external_input = args.dataset == 'neural_spike' or \
                        (args.dataset == 'mouse_head_direction' and args.experiment == 'no_hd_input')

    if with_hd_input:
        decoder = GNN_ENC_mousehd(num_neurons, args.history, args.hid_dim, args.dec_f_emb, args.activation)
    
    elif no_external_input:
        decoder = RNN_DEC_log(num_neurons, args.history, args.hid_dim, batch_size, 
                              args.dec_f_emb, args.dec_g1, args.g1_dim, args.activation)
    
    else:
        assert False, 'Invalid Experiment Configuration, \
            Try dataset + experiment for {mouse_hd_direction + with_hd_input / no_hd_input}  or\
            dataset for neural_spike'
    
    adj_inf_model, decoder = adj_inf_model.to(device), decoder.to(device)
    return adj_inf_model, decoder

def load_dataset(num_neurons, args, if_test, shuffle=True):
    history, pred_step, batch_size, rm_step = args.history, args.pred_step_p1, args.phase1_batchsize, args.removal_step
    neuron_type = args.neurons

    if args.dataset == 'neural_spike':
        if args.experiment == 'perturb_target':
            recording = 'pt'
        else:
            recording = 'eq'

        gts_featmat = np.load('data/spk_raw_{}_{}.npy'.format(recording, neuron_type))[:, :args.gts_totalstep]
        gts_featmat = gts_featmat.reshape(num_neurons, -1, args.bin).sum(axis=-1)

        if if_test:
            if args.experiment == 'perturb_target':
                testset = data_spike_poisson_rate(history, pred_step, 'test', binning=args.bin, 
                                                neuron_type=neuron_type, recording='pt', rm_step=rm_step, 
                                                decimated=False)
                edge_idx = testset.get_adjmat()
                test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) 
            elif args.experiment == 'perturb_decimation':
                testset = data_spike_poisson_rate(history, pred_step, 'test', binning=args.bin, 
                                                neuron_type=neuron_type, recording='eq', rm_step=rm_step, 
                                                decimated=True)
                edge_idx = testset.get_adjmat()
                test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) 
            else:
                testset = data_spike_poisson_rate(history, pred_step, 'test', binning=args.bin, 
                                                neuron_type=neuron_type, recording='eq', rm_step=rm_step, 
                                                decimated=False)
                edge_idx = testset.get_adjmat()
                test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) 
                
            return test_loader, gts_featmat, edge_idx
        
        else:
            if args.experiment == 'perturb_target':
                trainset = data_spike_poisson_rate(history, pred_step, 'train', binning=args.bin, 
                                                neuron_type=neuron_type, recording='pt', rm_step=rm_step, 
                                                decimated=False)
                validset = data_spike_poisson_rate(history, pred_step, 'valid', binning=args.bin, 
                                                neuron_type=neuron_type, recording='pt', rm_step=rm_step, 
                                                decimated=False)
                edge_idx = trainset.get_adjmat()
                train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle) 
                valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            elif args.experiment == 'perturb_decimation':
                trainset = data_spike_poisson_rate(history, pred_step, 'train', binning=args.bin, 
                                                neuron_type=neuron_type, recording='eq', rm_step=rm_step,
                                                decimated=True)
                validset = data_spike_poisson_rate(history, pred_step, 'valid', binning=args.bin, 
                                                neuron_type=neuron_type, recording='eq', rm_step=rm_step,
                                                decimated=True)
                edge_idx = trainset.get_adjmat()
                train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle) 
                valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            else:
                trainset = data_spike_poisson_rate(history, pred_step, 'train', binning=args.bin, 
                                                neuron_type=neuron_type, recording='eq', rm_step=rm_step, 
                                                decimated=False)
                validset = data_spike_poisson_rate(history, pred_step, 'valid', binning=args.bin, 
                                                neuron_type=neuron_type, recording='eq', rm_step=rm_step, 
                                                decimated=False)
                edge_idx = trainset.get_adjmat()
                train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
                valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
                
            return train_loader, valid_loader, gts_featmat, edge_idx
    
    elif args.dataset == 'mouse_head_direction':
        if not if_test:
            trainset = mouse_head_direction(history, pred_step, 'train')
            validset = mouse_head_direction(history, pred_step, 'valid')
            pref_HD, edge_idx = trainset.get_pref_HD(), trainset.get_adjmat()
            gts_featmat = np.load('data/mouse/gts_featmat.npy')
            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True)
            return train_loader, valid_loader, gts_featmat, pref_HD, edge_idx

        else:
            testset = mouse_head_direction(history, pred_step, 'test')
            pref_HD, edge_idx = testset.get_pref_HD(), testset.get_adjmat()
            gts_featmat = np.load('data/mouse/gts_featmat.npy')
            test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
            return test_loader, gts_featmat, pref_HD, edge_idx

    else:
        assert False, 'Invalid dataset, try {neural_spike or mouse_head_direction}'

def write_exp_log(file, args):
    if os.path.isfile(file):
        exp_log = open(file, 'a')
    else:
        exp_log = open(file, 'w')

    str_args = str(args)
    str_args = str_args.replace('Namespace(', '')
    str_args = str_args.replace(')', '')
    str_args = str_args.replace(',', '')
    str_args = str_args.replace(' ', ' --')
    str_args = str_args.replace('=', ' ')
    str_args = '--' + str_args + '\n'

    exp_log.write(str_args)
    exp_log.close()

def write_inf_log(file, logs):
    if os.path.isfile(file):
        exp_log = open(file, 'a')
    else:
        exp_log = open(file, 'w')

    str_inf_err = ''
    for item in logs:
        str_inf_err = str_inf_err + str(item) + '\n'

    exp_log.write(str_inf_err)
    exp_log.close()

def get_corr_by_rows(out, tar):
    corr_list = []
    for i in range(out.shape[0]):
        corr_single = np.corrcoef(out[i], tar[i])[1, 0]
        corr_list.append(corr_single)
    return corr_list

def decimation_process(args, ins, device):
    ins = torch.FloatTensor(ins).to(device)

    if args.remove_spike:
        ins = torch.poisson(ins)
        if args.mode == 'decim_manual':
            spk_rmvd, lams, R_iprs = decimation_method_iterX(ins, is_spike=True, is_manual=True, 
                                                             rm_step=args.removal_step)
        else:
            spk_rmvd, lams, R_iprs = decimation_method_iterX(ins, is_spike=True, is_manual=False)
        return spk_rmvd, lams, R_iprs

    else:
        if args.mode == 'decim_manual':
            lam_rmvd, lams, R_iprs = decimation_method_iterX(ins, is_spike=False, is_manual=True,
                                                             rm_step=args.removal_step)
        else:
            lam_rmvd, lams, R_iprs = decimation_method_iterX(ins, is_spike=False, is_manual=False)
        return lam_rmvd, lams, R_iprs