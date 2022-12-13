import numpy as np
import torch
import os
from generate_spike import random_configurations
from torch.utils.data import DataLoader
from model.nri_vsc_wo_pinputs import RNN_DEC_conv1D_lam
from utils import *

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--num_neurons', type=int, default=100)
parser.add_argument('--removal_step', type=int, default=50000)
parser.add_argument('--removal_mode', type=str, default='zeros') #zeros, spikes, removal
parser.add_argument('--sparsity', type=float, default=0.00625)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--mode', type=str, default='isi') #dec_eq, dec_pt
parser.add_argument('--neurons', type=str, default='lnp')
parser.add_argument('--perturb_steps', type=int, default='200')
parser.add_argument('--exp_num', type=str, default='1')
parser.add_argument('--exp_num_gen', type=str, default='1')
# Random Spike Generation
parser.add_argument('--num_batches', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--generation_step', type=int, default=48000)
# Decoder Setting
parser.add_argument('--hid_dim', type=int, default=32)
parser.add_argument('--g1_dim', type=int, default=10)
parser.add_argument('--history', type=int, default=200)

args = parser.parse_args()

num_neurons = args.num_neurons
neuron_type = args.neurons
rm_step = args.removal_step
rm_mode = args.removal_mode
p_spars = args.sparsity
pt_steps = args.perturb_steps
exp_num = args.exp_num
exp_num_gen = args.exp_num_gen

rand_nb = args.num_batches
rand_bs = args.batch_size
hid_dim = args.hid_dim
g1_dim = args.g1_dim
history = args.history
gen_step = args.generation_step

device = 'cuda:' + args.device
device = torch.device(device if torch.cuda.is_available() else 'cpu')

if neuron_type == 'binary':
    nt_str = neuron_type + str(history)
else:
    nt_str = 'LNP' + str(history)
save_folder = 'model/neural_spike/{}/perturbation/clam/complete/raw{}'.format(nt_str, exp_num)
plot_folder = 'plot/neural_spike/{}/perturbation/convlam/complete/raw{}'.format(nt_str, exp_num)
decoder_file = save_folder + '/dec_h{}_e.pt'.format(hid_dim)
z_file = plot_folder + '/z1_e.npy'

p_str = str(p_spars).replace('.', '')
if (rm_mode == 'zeros') or (rm_mode == 'removal'):
    p_str = '0'

if args.mode == 'dec_eq':
    srel = np.load('data/spk_raw_eq_lnp.npy')
    sreb = np.load('data/spk_raw_eq_binary.npy')

    if not os.path.exists('data/decimation'):
        os.makedirs('data/decimation')
        print('decimation(data) folder created')

    if not os.path.exists('fig/decimation'):
        os.makedirs('fig/decimation')
        print('decimation(fig) folder created')

    p_spars_list = [1/160, 0.01, 0.02, 0.05, 0.1, 0.2]

    for p_spars in p_spars_list:
        save_decimation(srel, rm_step, p_spars, 'spikes', 'lnp')
        save_decimation(sreb, rm_step, p_spars, 'spikes', 'binary')        

    save_decimation(srel, rm_step, 0, 'removal', 'lnp')
    save_decimation(srel, rm_step, 0, 'zeros', 'lnp')
    save_decimation(sreb, rm_step, 0, 'removal', 'binary')
    save_decimation(sreb, rm_step, 0, 'zeros', 'binary')

if args.mode == 'dec_pt':
    spk_pt = np.load('data/perturb/train_spk_{}_{}_{}.npy'.format(neuron_type, pt_steps, exp_num))
    num_neurons = spk_pt.shape[0]
    spk_pt = spk_pt.reshape(num_neurons, -1)

    if not os.path.exists('data/decimation'):
        os.makedirs('data/decimation')
        print('decimation(data) folder created')

    if not os.path.exists('fig/decimation'):
        os.makedirs('fig/decimation')
        print('decimation(fig) folder created')

    spk_rmvd, _ = decimation_method(spk_pt, rm_step, rm_mode, p_spars, device)
    spk_dec_file = '{}_{}_{}_{}.npy'.format(neuron_type, rm_mode, rm_step, p_str)

    if rm_mode == 'removal':
        isi, num_rmv, _, _ = check_isi(spk_rmvd)
        spk_rmvd = remove_zeros(spk_rmvd, num_rmv)
        spk_dec_file = 'data/decimation/{}_cprd_{}.npy'.format(neuron_type, rm_step)

    np.save(spk_dec_file, spk_rmvd)
    
    spike_plot(spk_rmvd, 100, 1, 'test')

if args.mode == 'eq':
    if not os.path.exists('data/decimation'):
        os.makedirs('data/decimation')
        print('decimation(data) folder created')

    if not os.path.exists('fig/decimation'):
        os.makedirs('fig/decimation')
        print('decimation(fig) folder created')

    spks = np.load('data/spk_raw_eq_{}.npy'.format(neuron_type))
    spks = spks*2 - 1
    lams = np.load('data/lam_raw_eq_{}.npy'.format(neuron_type))

    spks, lams = torch.FloatTensor(spks), torch.FloatTensor(lams)
    spks.to(device)
    lams.to(device)

    spks_rmvd, lam_rmvd, lam, R_ipr = decimation_method_iterX(spks, lams)
    spks_rmvd = spks_rmvd.detach().cpu().numpy()
    
    #np.save('data/decimation/spks_{}.npy'.format(args.mode), spks_rmvd)
    #np.save('data/decimation/pois_{}.npy'.format(args.mode), lam_rmvd)
    np.save('data/decimation/lams_{}.npy'.format(args.mode), lam)
    np.save('data/decimation/Ripr_{}.npy'.format(args.mode), R_ipr)

    print(R_ipr)
    spike_plot(spks_rmvd, 100, 1, 'eq_{}_{}'.format(neuron_type, exp_num))

if args.mode == 'random_uniform':
    decoder = RNN_DEC_conv1D_lam(hid_dim, g1_dim, hid_dim, rand_bs, history, num_neurons)
    decoder.to(device)
    decoder.load_state_dict(torch.load(decoder_file, map_location=device))
    z = np.load(z_file)
    z = torch.FloatTensor(z).to(device)

    dataset = random_configurations(p=0.5, num_neurons=100, history=200, num_batches=rand_nb, batch_size=rand_bs)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    lam_list, spk_list = [], []
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader)):
            x_spk, y_lam, x_lam, edge_idx = data
            x_spk, y_lam, x_lam, edge_idx = (x_spk.squeeze(), y_lam.squeeze(), 
                                             x_lam.reshape(num_neurons, rand_bs, -1), edge_idx.squeeze())
            x_spk, y_lam, x_lam, edge_idx = (x_spk.to(device), y_lam.to(device), 
                                             x_lam.to(device), edge_idx.to(device))

            out_lam = decoder(x_spk, edge_idx, z, x_lam, gen_step)
            out_spk = torch.poisson(out_lam)

            lam_list.append(out_lam.cpu().detach().numpy())
            spk_list.append(out_spk.cpu().detach().numpy())
    
    lam_np = np.stack(lam_list, axis=1).reshape(num_neurons, -1, gen_step)
    spk_np = np.stack(spk_list, axis=1).reshape(num_neurons, -1, gen_step)

    if not os.path.exists('data/rand'):
        os.makedirs('data/rand')
    
    np.save('data/rand/uniform_lam.npy', lam_np)
    np.save('data/rand/uniform_spk.npy', spk_np)

    print(spk_np[:,0,:].shape)
    spike_plot(spk_np[:,0,:], 100, 1, 'rnd_uniform_{}_{}'.format(neuron_type, exp_num_gen))

if args.mode == 'random_isi':
    decoder = RNN_DEC_conv1D_lam(hid_dim, g1_dim, hid_dim, rand_bs, history, num_neurons)
    decoder.to(device)
    decoder.load_state_dict(torch.load(decoder_file, map_location=device))
    z = np.load(z_file)
    z = torch.FloatTensor(z).to(device)

    dataset = random_configurations(p=p_spars, num_neurons=100, history=200, num_batches=rand_nb, batch_size=rand_bs)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    lam_list, spk_list = [], []
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader)):
            x_spk, y_lam, x_lam, edge_idx = data
            x_spk, y_lam, x_lam, edge_idx = (x_spk.squeeze(), y_lam.squeeze(), 
                                             x_lam.reshape(num_neurons, rand_bs, -1), edge_idx.squeeze())
            x_spk, y_lam, x_lam, edge_idx = (x_spk.to(device), y_lam.to(device), 
                                             x_lam.to(device), edge_idx.to(device))

            out_lam = decoder(x_spk, edge_idx, z, x_lam, gen_step)
            out_spk = torch.poisson(out_lam)

            lam_list.append(out_lam.cpu().detach().numpy())
            spk_list.append(out_spk.cpu().detach().numpy())
    
    lam_np = np.stack(lam_list, axis=1).reshape(num_neurons, -1, gen_step)
    spk_np = np.stack(spk_list, axis=1).reshape(num_neurons, -1, gen_step)

    if not os.path.exists('data/rand'):
        os.makedirs('data/rand')
    
    np.save('data/rand/isi_lam.npy', lam_np)
    np.save('data/rand/isi_spk.npy', spk_np)

    print(spk_np[:,0,:].shape)
    spike_plot(spk_np[:,0,:], 100, 1, 'rnd_isi_{}_{}'.format(neuron_type, exp_num_gen))

if args.mode == 'plot':
    if not os.path.exists('data/decimation'):
        os.makedirs('data/decimation')
        print('decimation(data) folder created')

    if not os.path.exists('fig/decimation'):
        os.makedirs('fig/decimation')
        print('decimation(fig) folder created')
        
    spk_isi = np.load('data/rand/isi_spk.npy')
    spk_uniform = np.load('data/rand/uniform_spk.npy')
    spk_eq = np.load('data/spk_raw_eq_lnp.npy')

    spk_isi, spk_uniform = spk_isi.squeeze(), spk_uniform.squeeze()

    inv_isi = spk_isi*2 - 1
    inv_uniform = spk_isi*2 - 1
    inv_eq = spk_eq*2 - 1

    n_steps = spk_isi.shape[1]

    
    C_si, C_su, C_se = (np.matmul(spk_isi, spk_isi.T)/n_steps, np.matmul(spk_uniform, spk_uniform.T)/n_steps, 
                        np.matmul(spk_eq, spk_eq.T)/n_steps)
    C_ii, C_iu, C_ie = (np.matmul(inv_isi, inv_isi.T)/n_steps, np.matmul(inv_uniform, inv_uniform.T)/n_steps, 
                        np.matmul(inv_eq, inv_eq.T)/n_steps)

    C_si, C_su, C_se = torch.FloatTensor(C_si), torch.FloatTensor(C_su), torch.FloatTensor(C_se)
    C_ii, C_iu, C_ie = torch.FloatTensor(C_ii), torch.FloatTensor(C_iu), torch.FloatTensor(C_ie)

    R_si, lam_si = get_R_ipr(C_si)
    R_su, lam_su = get_R_ipr(C_su)
    R_se, lam_se = get_R_ipr(C_se)
    R_ii, lam_ii = get_R_ipr(C_ii)
    R_iu, lam_iu = get_R_ipr(C_iu)
    R_ie, lam_ie = get_R_ipr(C_ie)

    lam_list_s = [lam_si.cpu().detach().numpy(), lam_su.cpu().detach().numpy(), lam_se.cpu().detach().numpy()]
    lam_list_i = [lam_ii.cpu().detach().numpy(), lam_iu.cpu().detach().numpy(), lam_ie.cpu().detach().numpy()]

    label_list = ['Original', 'Random(uniform)', 'Random(ISI)']
    plot_lamlist(lam_list_s, label_list, suffix='raw_s')
    plot_lamlist(lam_list_i, label_list, suffix='raw_i')

    print(R_si, R_su, R_se)
    print(R_ii, R_iu, R_ie)