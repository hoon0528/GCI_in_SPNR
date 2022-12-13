import time
import os
import argparse

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import yaml
from utils import *

#random.seed(42)
torch.manual_seed(42)
#torch.cuda.manual_seed(42)
#np.random.seed(42)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

##################################### Argument ##############################################
parser = argparse.ArgumentParser()
# Dataset Configuration
parser.add_argument('--dataset', type=str, default='mouse_head_direction')
parser.add_argument('--neurons', type=str, default='lnp') # binary/lnp
parser.add_argument('--bin', type=int, default=1)
# Experiments Configuration
parser.add_argument('--experiment', type=str, default='with_hd_input') # equilibrium or decimation / with_hd_input or no_hd_input
parser.add_argument('--reconstruction', type=int, default=0)
parser.add_argument('--loss_scaler', type=float, default=30)
parser.add_argument('--history', type=int, default=200)
parser.add_argument('--pred_step_p1', type=int, default=20) # 20
parser.add_argument('--msg_hop', type=int, default=2) # 1 2 4 8
parser.add_argument('--perturb_step', type=int, default=200)
parser.add_argument('--decoder', type=str, default='log') #log lam
parser.add_argument('--dec_f_emb', type=str, default='mlp') #conv mlp
parser.add_argument('--dec_g1', type=str, default='identity') #identity mlp
parser.add_argument('--activation', type=str, default='tanh') #tanh, sigmoid, relu
parser.add_argument('--z_act', type=str, default='identity') #tanh, sigmoid, relu
parser.add_argument('--if_symmode', type=int, default=0)
parser.add_argument('--symmode', type=str, default='ltmatonly')
parser.add_argument('--add_similarity_z_term', type=int, default=1)
parser.add_argument('--similarity_z_term', type=str, default='cos')
parser.add_argument('--gts_totalstep', type=int, default=3840000) #1280000
parser.add_argument('--hidden_rand_init', type=int, default=1)
parser.add_argument('--removal_step', type=int, default=1500000)
# Dimension of Layers
parser.add_argument('--hid_dim', type=int, default=32)
parser.add_argument('--out_channel', type=int, default=10) # Set 10
parser.add_argument('--g1_dim', type=int, default=10) # 1, 10
parser.add_argument('--kernal_x_1', type=int, default=200)
parser.add_argument('--kernal_x_2', type=int, default=200)
parser.add_argument('--stride_x_1', type=int, default=20)
parser.add_argument('--stride_x_2', type=int, default=20)
# Learning features
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--phase1_batchsize', type=int, default=5)
parser.add_argument('--phase2_batchsize', type=int, default=5)
parser.add_argument('--device', type=str, default=0)
# Exp. Num
parser.add_argument('--exp_num', type=str, default=0)

############################ Hyperparams Configuration ######################################
args = parser.parse_args()

exp_log_file = 'exp_log_{}.txt'.format(args.dataset)

with open('config/{}.yml'.format(args.dataset), encoding='UTF8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

exp_config = config.get('experiment')
tr_config = config.get('training')
model_config = config.get('model')

num_nodes, neuron_type, args.gts_totalstep = exp_config.get('num_nodes'), args.neurons, exp_config.get('tr_length')
history, pred_step_p1, msg_hop, rm_step = args.history, args.pred_step_p1, args.msg_hop, args.removal_step

lr, lr_decay, gamma, epochs = tr_config.get('lr'), tr_config.get('lr_decay'), tr_config.get('gamma'), args.epochs
p1_bs, p2_bs, perturb_step = args.phase1_batchsize, args.phase2_batchsize, args.perturb_step
enc_hid = dec_msg_hid = dec_msg_out = dec_hid = args.hid_dim

with_hd_input = args.dataset == 'mouse_head_direction' and args.experiment == 'with_hd_input'
no_external_input = args.dataset == 'neural_spike' or \
                    (args.dataset == 'mouse_head_direction' and args.experiment == 'no_hd_input')

device = 'cuda:' + str(args.device)
device = torch.device(device if torch.cuda.is_available() else 'cpu')

########################## Set directories of trained models ###############################
save_folder, model_directory = save_directory(args)

if args.experiment == 'perturb_joint':
    encoder1_file, encoder2_file = model_directory[0], model_directory[1]
    decoder1_file, decoder2_file = model_directory[2], model_directory[3]
    loss_file, log_file = model_directory[4], model_directory[5]
    z1_file, z2_file = model_directory[6], model_directory[7]
else:
    encoder_file, decoder_file = model_directory[0], model_directory[1]
    loss_file, log_file, z_file = model_directory[2], model_directory[3], model_directory[4]

print('{} {} {} DEC:{} {} {}'.format(args.neurons, args.experiment, args.bin, 
                                     args.decoder + args.dec_f_emb, args.dataset, args.exp_num))

if os.path.isfile(log_file):
    log = open(log_file, 'a')
else:
    log = open(log_file, 'w')

# Define models & Load dataset
if args.dataset == 'neural_spike':
    train_loader, valid_loader, gts_featmat, adj_mat = load_dataset(num_nodes, args, if_test=False)
    
elif args.dataset == 'mouse_head_direction':
    train_loader, valid_loader, gts_featmat, pref_head_direction, adj_mat = \
        load_dataset(num_nodes, args, if_test=False)
    
else:
    assert False, 'Invalid dataset, try neural_spike or mouse_head_direction'
adj_inf_model, decoder = load_models(num_nodes, args, device)

# Optimizer setting
optimizer = optim.Adam(list(adj_inf_model.parameters()) + list(decoder.parameters()), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

# Target weight profile of Neural Circuit
weight_profile = np.load('data/connectivity_W100.npy')
np.fill_diagonal(weight_profile, 0)
weight_profile = weight_profile[~np.eye(weight_profile.shape[0],dtype=bool)].reshape(-1, 1)

################################### Training ##############################################
def train_perturb(epoch, best_val_loss):
    t = time.time()
    
    cos_tar = torch.ones(1)
    cos_tar = cos_tar.to(device)
    
    gts_input = torch.FloatTensor(gts_featmat)
    gts_input = gts_input.to(device)
    
    poisson_p1_train = []
    poisson_p2_train = []
    z_loss_train = []

    adj_inf_model.train()
    decoder.train()

    adj_inf_model.to(device)
    decoder.to(device)
    decoder.to(device)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        x_spk, tar_spk, edge_idx = data
        x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
        x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)

        optimizer.zero_grad()

        hidden_enc, z1_corr = adj_inf_model(gts_input, edge_idx)
        z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
        z1 = -torch.sigmoid(z1_corr)
        z1 = z1.reshape(-1, 1)

        out_p1 = decoder(x_spk, edge_idx, z1, hidden_enc, pred_step_p1)
        loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

        spike_p2 = []
        with torch.no_grad():
            spk_p2_init = x_spk[:, :, :history]
            decoder.load_state_dict(decoder.state_dict())

            for i in range(perturb_step):
                if i == 0:
                    lam_p2 = decoder(spk_p2_init, edge_idx, z1, 1) #[100, 80, 200] > [100, 80, 1, 1]
                    spk_p2 = torch.poisson(lam_p2.exp()) #cat [100, 80, 1, 1]
                    spk_cat = spk_p2.reshape(num_nodes, p1_bs, 1)
                    spk_next = torch.cat((spk_p2_init[:, :, -199:], spk_cat), dim=-1)
                else:
                    lam_p2 = decoder(spk_next, edge_idx, z1, 1) #[100, 80, 200] > [100, 80, 1, 1]
                    spk_p2 = torch.poisson(lam_p2.exp()) #cat [100, 80, 1, 1]
                    spk_cat = spk_p2.reshape(num_nodes, p1_bs, 1)
                    spk_next = torch.cat((spk_next[:, :, -199:], spk_cat), dim=-1)
            spike_p2.append(spk_next)

            spike_p2 = torch.stack(spike_p2, dim=2) #[nodes, p1_bs, perturb_step, feat_dim]
            spike_p2 = spike_p2.reshape(num_nodes, -1) #[nodes, p1_bs*perturb_step]
            x_spk_p2, tar_spk_p2 = to_dec_batch(spike_p2, num_nodes, history, pred_step_p2, p2_bs)
            
            p2_step = spike_p2.shape[1]
            count_p2 = int(args.gts_totalstep / p2_step)
            spike_p2_featmat = torch.unsqueeze(spike_p2, dim=1)
            spike_p2_featmat = spike_p2_featmat.expand(num_nodes, count_p2+1, p2_step)
            spike_p2_featmat = spike_p2_featmat.reshape(num_nodes, -1)
            spike_p2_featmat = spike_p2_featmat[:, :args.gts_totalstep]

        _, z2_corr = adj_inf_model(spike_p2_featmat, edge_idx)
        z2_corr = make_z_sym_gts(z2_corr, num_nodes, device, args.symmode)
        z2 = -torch.sigmoid(z2_corr)
        z2 = z2.reshape(-1, 1)

        if args.similarity_z_term == 'cos':
            loss_z_similar = F.cosine_embedding_loss(z1.t(), z2.t(), cos_tar)
        elif args.similarity_z_term == 'none':
            loss_z_similar = 0
        else:
            assert False, 'Invalid z loss function'  

        out_p2_list = []
        for i in range(x_spk_p2.shape[1]):
            out_p2 = decoder(x_spk_p2[:, i, :, :], edge_idx, z2, pred_step_p2) #[100, p2_bs, 20, 1]
            out_p2_list.append(out_p2.squeeze())
        out_p2_aggr = torch.stack(out_p2_list, dim=1)
        loss_recon_p2 = F.poisson_nll_loss(out_p2_aggr, tar_spk_p2, log_input=True)

        if args.add_similarity_z_term:
            loss = loss_recon_p1 + loss_recon_p2 + args.loss_scaler*loss_z_similar
        else:
            loss = loss_recon_p1 + loss_recon_p2
        loss.backward()
        optimizer.step()

        poisson_p1_train.append(loss_recon_p1.item())
        poisson_p2_train.append(loss_recon_p2.item())
        z_loss_train.append(loss_z_similar.item())
        
    scheduler.step()

    poisson_p1_valid = []
    poisson_p2_valid = []
    z_loss_valid = []

    adj_inf_model.eval()
    decoder.eval()
    
    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            x_spk, tar_spk, edge_idx = data
            x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
            x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)
            
            optimizer.zero_grad()

            hidden_enc, z1_corr = adj_inf_model(gts_input, edge_idx)
            z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
            z1 = -torch.sigmoid(z1_corr)
            z1 = z1.reshape(-1, 1)

            out_p1 = decoder(x_spk, edge_idx, z1, hidden_enc, pred_step_p1)
            loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

            spike_p2 = []
            with torch.no_grad():
                spk_p2_init = x_spk[:, :, :history]
                decoder.load_state_dict(decoder.state_dict())
            
                for i in range(perturb_step):
                    if i == 0:
                        lam_p2 = decoder(spk_p2_init, edge_idx, z1, 1) #[100, 80, 200] > [100, 80, 1, 1]
                        spk_p2 = torch.poisson(lam_p2.exp()) #cat [100, 80, 1, 1]
                        spk_cat = spk_p2.reshape(num_nodes, p1_bs, 1)
                        spk_next = torch.cat((spk_p2_init[:, :, -199:], spk_cat), dim=-1)
                    else:
                        lam_p2 = decoder(spk_next, edge_idx, z1, 1) #[100, 80, 200] > [100, 80, 1, 1]
                        spk_p2 = torch.poisson(lam_p2.exp()) #cat [100, 80, 1, 1]
                        spk_cat = spk_p2.reshape(num_nodes, p1_bs, 1)
                        spk_next = torch.cat((spk_next[:, :, -199:], spk_cat), dim=-1)
                spike_p2.append(spk_next)

                spike_p2 = torch.stack(spike_p2, dim=2) #[nodes, p1_bs, perturb_step, feat_dim]
                spike_p2 = spike_p2.reshape(num_nodes, -1) #[nodes, p1_bs*perturb_step]
                x_spk_p2, tar_spk_p2 = to_dec_batch(spike_p2, num_nodes, history, pred_step_p2, p2_bs)

                p2_step = spike_p2.shape[1]
                count_p2 = int(args.gts_totalstep / p2_step)
                spike_p2_featmat = torch.unsqueeze(spike_p2, dim=1)
                spike_p2_featmat = spike_p2_featmat.expand(num_nodes, count_p2+1, p2_step)
                spike_p2_featmat = spike_p2_featmat.reshape(num_nodes, -1)
                spike_p2_featmat = spike_p2_featmat[:, :args.gts_totalstep]

            _, z2_corr = adj_inf_model(spike_p2_featmat, edge_idx)
                
            z2_corr = make_z_sym_gts(z2_corr, num_nodes, device, args.symmode)
            z2 = -torch.sigmoid(z2_corr)
            z2 = z2.reshape(-1, 1)

            if args.similarity_z_term == 'cos':
                loss_z_similar = F.cosine_embedding_loss(z1.t(), z2.t(), cos_tar)
            elif args.similarity_z_term == 'none':
                loss_z_similar = 0
            else:
                assert False, 'Invalid z loss function'  

            out_p2_list = []
            for i in range(x_spk_p2.shape[1]):
                out_p2 = decoder(x_spk_p2[:, i, :, :], edge_idx, z2, pred_step_p2) #[100, p2_bs, 20, 1]
                out_p2_list.append(out_p2.squeeze())
            out_p2_aggr = torch.stack(out_p2_list, dim=1)
            loss_recon_p2 = F.poisson_nll_loss(out_p2_aggr, tar_spk_p2, log_input=True)

            if args.add_similarity_z_term:
                loss = loss_recon_p1 + loss_recon_p2 + args.loss_scaler*loss_z_similar
            else:
                loss = loss_recon_p1 + loss_recon_p2

            poisson_p1_valid.append(loss_recon_p1.item())
            poisson_p2_valid.append(loss_recon_p2.item())
            z_loss_valid.append(loss_z_similar.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
              'pois_p2_train: {:.10f}'.format(np.mean(poisson_p2_train)),
              'z_cos_train: {:.10f}'.format(np.mean(z_loss_train)),
              'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
              'pois_p2_val: {:.10f}'.format(np.mean(poisson_p2_valid)),
              'z_cos_val: {:.10f}'.format(np.mean(z_loss_valid)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(poisson_p1_valid) < best_val_loss:
            torch.save(adj_inf_model.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder1_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
                  'pois_p2_train: {:.10f}'.format(np.mean(poisson_p2_train)),
                  'z_cos_train: {:.10f}'.format(np.mean(z_loss_train)),
                  'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
                  'pois_p2_val: {:.10f}'.format(np.mean(poisson_p2_valid)),
                  'z_cos_val: {:.10f}'.format(np.mean(z_loss_valid)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
        
        loss = np.array([np.mean(poisson_p1_train), np.mean(poisson_p2_train), np.mean(z_loss_train), 
                         np.mean(poisson_p1_valid), np.mean(poisson_p2_valid), np.mean(z_loss_valid)])
    return np.mean(poisson_p1_valid), loss

def train_eq_only(epoch, best_val_loss):
    t = time.time()
      
    gts_input = torch.FloatTensor(gts_featmat)
    gts_input = gts_input.to(device)
    edge_idx, pref_HD = adj_mat, pref_head_direction

    poisson_p1_train = []

    adj_inf_model.train()
    decoder.train()

    adj_inf_model.to(device)
    decoder.to(device)

    for _, data in tqdm(enumerate(train_loader)):
        if args.dataset == 'neural_spike':
            x_spk, tar_spk = data
            x_spk, tar_spk = x_spk.to(device), tar_spk.to(device)
            edge_idx = edge_idx.to(device)

        elif args.dataset == 'mouse_head_direction':
            x_spk, tar_spk, ang = data
            x_spk, tar_spk, ang = x_spk.to(device), tar_spk.to(device), ang.to(device)
            edge_idx, pref_HD = edge_idx.to(device), pref_HD.to(device)
        
        else:
            assert False, 'Invalid dataset, Try neural_spike or mouse_head_direction'

        tar_spk = tar_spk.transpose(0, 1)

        optimizer.zero_grad()

        _, z1_corr = adj_inf_model(gts_input, edge_idx)
        if args.if_symmode:
            edge_idx, z1 = make_z_sym_gts(z1_corr, edge_idx, num_nodes, args.symmode)
        else:
            z1 = z1_corr

        if args.z_act == 'sigmoid':
            z1 = -torch.sigmoid(z1)
        elif args.z_act == 'tanh':
            z1 = torch.tanh(z1)
        z1 = z1.reshape(-1, 1)

        #Poisson Loss
        if with_hd_input:
            if args.reconstruction:
                out_p1, _ = decoder(x_spk, ang, edge_idx, z1, pref_HD, True, pred_step_p1, msg_hop)
                loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

            else:
                out_p1, _ = decoder(x_spk, ang, edge_idx, z1, pref_HD, False, pred_step_p1, msg_hop)
                loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)
        
        elif no_external_input:
            if args.reconstruction:
                out_p1 = decoder(x_spk, edge_idx, z1, True, pred_step_p1, msg_hop)
                loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

            else:
                out_p1 = decoder(x_spk, edge_idx, z1, False, pred_step_p1, msg_hop)
                loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

        loss_recon_p1.backward()
        optimizer.step()

        poisson_p1_train.append(loss_recon_p1.item())
        
    scheduler.step()
    poisson_p1_valid = []

    adj_inf_model.eval()
    decoder.eval()
    
    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            if args.dataset == 'neural_spike':
                x_spk, tar_spk = data
                x_spk, tar_spk = x_spk.to(device), tar_spk.to(device)
                edge_idx = edge_idx.to(device)

            elif args.dataset == 'mouse_head_direction':
                x_spk, tar_spk, ang = data
                x_spk, tar_spk, ang = x_spk.to(device), tar_spk.to(device), ang.to(device)
                edge_idx, pref_HD = edge_idx.to(device), pref_HD.to(device)
            
            else:
                assert False, 'Invalid dataset, Try neural_spike or mouse_head_direction'

            tar_spk = tar_spk.transpose(0, 1)
            
            optimizer.zero_grad()

            _, z1_corr = adj_inf_model(gts_input, edge_idx)
            if args.if_symmode:
                edge_idx, z1 = make_z_sym_gts(z1_corr, edge_idx, num_nodes, args.symmode)
            else:
                z1 = z1_corr

            if args.z_act == 'sigmoid':
                z1 = -torch.sigmoid(z1_corr)
            elif args.z_act == 'tanh':
                z1 = torch.tanh(z1_corr)
            z1 = z1.reshape(-1, 1)

            if with_hd_input:
                if args.reconstruction:
                    out_p1, _ = decoder(x_spk, ang, edge_idx, z1, pref_HD, True, pred_step_p1, msg_hop)
                    loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

                else:
                    out_p1, _ = decoder(x_spk, ang, edge_idx, z1, pref_HD, False, pred_step_p1, msg_hop)
                    loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)
            
            elif no_external_input:
                if args.reconstruction:
                    out_p1 = decoder(x_spk, edge_idx, z1, True, pred_step_p1, msg_hop)
                    loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

                else:
                    out_p1 = decoder(x_spk, edge_idx, z1, False, pred_step_p1, msg_hop)
                    loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

            poisson_p1_valid.append(loss_recon_p1.item())

        z_item = z1.clone().cpu().detach().numpy()
        
        print('Epoch: {:04d}'.format(epoch),
              'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
              'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(poisson_p1_valid) < best_val_loss:
            torch.save(adj_inf_model.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
                  'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
        
        loss = np.array([np.mean(poisson_p1_train), np.mean(poisson_p1_valid)])
    return np.mean(poisson_p1_valid), loss, z_item

def train_perturb_target(epoch, best_val_p1, best_val_p2):
    t = time.time()
    
    cos_tar = torch.ones(1)
    cos_tar = cos_tar.to(device)
    
    gts_input = torch.FloatTensor(gts_featmat)
    gts_input = gts_input.to(device)
    
    per_input_all = np.load('data/spk_bin_per.npy')
    per_input_all = torch.FloatTensor(per_input_all)
    per_input_train = per_input_all[:, :480000]
    per_input_valid = per_input_all[:, 480000:528000]
    
    x2_input_tr, x2_target_tr = to_dec_batch_ptar(per_input_train, num_nodes, history, pred_step_p1, p2_bs)
    x2_input_val, x2_target_val = to_dec_batch_ptar(per_input_valid, num_nodes, history, pred_step_p2, p2_bs)
    rand_idx_tr = torch.randperm(x2_input_tr.shape[1])
    rand_idx_val = torch.randperm(x2_input_val.shape[1])
    x2_input_tr, x2_target_tr = x2_input_tr[:, rand_idx_tr, :, :], x2_target_tr[:, rand_idx_tr, :, :]
    x2_input_val, x2_target_val = x2_input_val[:, rand_idx_val, :, :], x2_target_val[:, rand_idx_val, :, :]

    per_input = per_input_all[:, :args.gts_totalstep]
    per_input, x2_input_tr, x2_target_tr = per_input.to(device), x2_input_tr.to(device), x2_target_tr.to(device)
    x2_input_val, x2_target_val = x2_input_val.to(device), x2_target_val.to(device)
    
    poisson_p1_train = []
    poisson_p2_train = []
    z_loss_train = []

    adj_inf_model.train()
    decoder.train()

    adj_inf_model.to(device)
    decoder.to(device)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        x_spk, tar_spk, edge_idx = data
        x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
        x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)

        optimizer.zero_grad()

        hidden_enc, z1_corr = adj_inf_model(gts_input, edge_idx)
        z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
        z1 = -torch.sigmoid(z1_corr)
        z1 = z1.reshape(-1, 1)

        out_p1 = decoder(x_spk, edge_idx, z1, hidden_enc, pred_step_p1)
        loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

        _, z2_corr = adj_inf_model(per_input, edge_idx)
        z2_corr = make_z_sym_gts(z2_corr, num_nodes, device, args.symmode)
        z2 = -torch.sigmoid(z2_corr)
        z2 = z2.reshape(-1, 1)

        if args.similarity_z_term == 'cos':
            loss_z_similar = F.cosine_embedding_loss(z1.t(), z2.t(), cos_tar)
        elif args.similarity_z_term == 'none':
            loss_z_similar = 0
        else:
            assert False, 'Invalid z loss function'  

        out_p2 = decoder(x2_input_tr[:, batch_idx, :, :], edge_idx, z2, pred_step_p2) #[100, p2_bs, 20, 1]
        loss_recon_p2 = F.poisson_nll_loss(out_p2.squeeze(), x2_target_tr[:, batch_idx, :, :], log_input=True)

        if args.add_similarity_z_term:
            loss = loss_recon_p1 + loss_recon_p2 + args.loss_scaler*loss_z_similar
        else:
            loss = loss_recon_p1 + loss_recon_p2
        loss.backward()
        optimizer.step()

        poisson_p1_train.append(loss_recon_p1.item())
        poisson_p2_train.append(loss_recon_p2.item())
        z_loss_train.append(loss_z_similar.item())
        
    scheduler.step()

    poisson_p1_valid = []
    poisson_p2_valid = []
    z_loss_valid = []

    adj_inf_model.eval()
    decoder.eval()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            x_spk, tar_spk, edge_idx = data
            x_spk, tar_spk, edge_idx = x_spk.squeeze(), tar_spk.squeeze(), edge_idx.squeeze()
            x_spk, tar_spk, edge_idx = x_spk.to(device), tar_spk.to(device), edge_idx.to(device)
            
            optimizer.zero_grad()

            hidden_enc, z1_corr = adj_inf_model(gts_input, edge_idx)
            z1_corr = make_z_sym_gts(z1_corr, num_nodes, device, args.symmode)
            z1 = -torch.sigmoid(z1_corr)
            z1 = z1.reshape(-1, 1)

            out_p1 = decoder(x_spk, edge_idx, z1, hidden_enc, pred_step_p1)
            loss_recon_p1 = F.poisson_nll_loss(out_p1.squeeze(), tar_spk, log_input=True)

            _, z2_corr = adj_inf_model(per_input, edge_idx)
                
            z2_corr = make_z_sym_gts(z2_corr, num_nodes, device, args.symmode)
            z2 = -torch.sigmoid(z2_corr)
            z2 = z2.reshape(-1, 1)

            if args.similarity_z_term == 'cos':
                loss_z_similar = F.cosine_embedding_loss(z1.t(), z2.t(), cos_tar)
            elif args.similarity_z_term == 'none':
                loss_z_similar = 0
            else:
                assert False, 'Invalid z loss function'  

            out_p2 = decoder(x2_input_val[:, batch_idx, :, :], edge_idx, z2, pred_step_p2) #[100, p2_bs, 20, 1]
            loss_recon_p2 = F.poisson_nll_loss(out_p2.squeeze(), x2_target_val[:, batch_idx, :, :], log_input=True)

            if args.add_similarity_z_term:
                loss = loss_recon_p1 + loss_recon_p2 + args.loss_scaler*loss_z_similar
            else:
                loss = loss_recon_p1 + loss_recon_p2

            poisson_p1_valid.append(loss_recon_p1.item())
            poisson_p2_valid.append(loss_recon_p2.item())
            z_loss_valid.append(loss_z_similar.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
              'pois_p2_train: {:.10f}'.format(np.mean(poisson_p2_train)),
              'z_cos_train: {:.10f}'.format(np.mean(z_loss_train)),
              'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
              'pois_p2_val: {:.10f}'.format(np.mean(poisson_p2_valid)),
              'z_cos_val: {:.10f}'.format(np.mean(z_loss_valid)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(poisson_p1_valid) < best_val_p1:
            torch.save(adj_inf_model.state_dict(), encoder1_file)
            torch.save(decoder.state_dict(), decoder1_file)
            print('Best equilibrium model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'pois_p1_train: {:.10f}'.format(np.mean(poisson_p1_train)),
                  'pois_p2_train: {:.10f}'.format(np.mean(poisson_p2_train)),
                  'z_cos_train: {:.10f}'.format(np.mean(z_loss_train)),
                  'pois_p1_val: {:.10f}'.format(np.mean(poisson_p1_valid)),
                  'pois_p2_val: {:.10f}'.format(np.mean(poisson_p2_valid)),
                  'z_cos_val: {:.10f}'.format(np.mean(z_loss_valid)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()

        if np.mean(poisson_p2_valid) < best_val_p2:
            torch.save(adj_inf_model.state_dict(), encoder2_file)
            torch.save(decoder.state_dict(), decoder2_file)
            print('Best perturbation model so far, saving...')

        loss = np.array([np.mean(poisson_p1_train), np.mean(poisson_p2_train), np.mean(z_loss_train), 
                         np.mean(poisson_p1_valid), np.mean(poisson_p2_valid), np.mean(z_loss_valid)])
    return np.mean(poisson_p1_valid), np.mean(poisson_p2_valid), loss

z_list = []
loss_list = []
best_val_loss = np.inf
best_val_p1, best_val_p2 = np.inf, np.inf
best_epoch = 0

for epoch in tqdm(range(epochs)):
    poiss_p1_val, loss, z = train_eq_only(epoch, best_val_loss)
    loss_list.append(loss)
    z_list.append(z)
    if poiss_p1_val < best_val_loss:
        best_val_loss = poiss_p1_val
        best_epoch = epoch

np.save(loss_file, np.stack(loss_list, axis=0))
np.save(z_file, np.stack(z_list, axis=0))
write_exp_log(file=exp_log_file, args=args)