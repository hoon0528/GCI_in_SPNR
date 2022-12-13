''' NRI style Decoder '''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from SourceModified.message_passing import MessagePassing2

import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class RNN_DEC_lam(MessagePassing):
    def __init__(self, nodes, history, n_hid, batch_size, f_emb, g1, g1_out, act, do_prob=0.):
        super(RNN_DEC_lam, self).__init__(aggr='add') # Eq 14
        self.hidden_dim = n_hid
        self.out_channel = n_hid
        self.batchsize = batch_size
        self.do_prob = do_prob
        self.nodes = nodes
        self.edges = nodes * (nodes-1)
        self.history = history
        
        self.f_emb, self.is_emb_conv = self.define_f_emb(n_in=history, n_hid=n_hid, function=f_emb, activation=act)
        self.g1, g1_dim = self.define_g1(g1_in=n_hid, g1_out=g1_out, function=g1, activation=act)
        self.gru_cell = nn.GRUCell(g1_dim+n_hid, n_hid)
        
        self.fc1_hat_e = nn.Linear(2 * n_hid, n_hid) # fc1 to fe
        self.fc2_hat_e = nn.Linear(n_hid, n_hid)

        self.fc1_out = nn.Linear(n_hid, n_hid)
        self.fc2_out = nn.Linear(n_hid, 1)

    def define_f_emb(self, n_in, n_hid, function, activation):
        f_emb = nn.Sequential()

        if function == 'conv':
            f_emb.add_module('f_emb', nn.Conv1d(1, n_hid, n_in))
            is_emb_conv = True
        elif function == 'mlp':
            f_emb.add_module('f_emb', nn.Linear(n_in, n_hid))
            is_emb_conv = False
        else:
            assert False, 'Invalid emb function type, try {mlp or conv}'

        if activation == 'sigmoid':
            f_emb.add_module('f_emb_act', nn.Sigmoid())
        elif activation == 'tanh':
            f_emb.add_module('f_emb_act', nn.Tanh())
        elif activation == 'relu':
            f_emb.add_module('f_emb_act', nn.ReLU())
        else:
            assert False, 'Invalid activation function type, try {sigmoid, tanh or relu}'
        
        return f_emb, is_emb_conv

    def define_g1(self, g1_in, g1_out, function, activation):
        g1 = nn.Sequential()

        if function == 'identity':
            g1_out = g1_in
            g1.add_module('g1', nn.Identity())
        elif function == 'mlp':
            g1.add_module('g1', nn.Linear(g1_in, g1_out))
        else:
            assert False, 'Invalid g1 function type, try {identity, mlp or }'

        if function != 'identity':
            if activation == 'sigmoid':
                g1.add_module('g1_act', nn.Sigmoid())
            elif activation == 'tanh':
                g1.add_module('g1_act', nn.Tanh())
            elif activation == 'relu':
                g1.add_module('g1_act', nn.ReLU())
            else:
                assert False, 'Invalid activation function type, try {sigmoid, tanh or relu}'
        
        return g1, g1_out
        
    def forward(self, inputs, edge_index, z, lam_t, reconstruction=False, pred_steps=1):
        # [num_nodes, phase1_batch, tsteps]
        # num_timesteps should be [window_size = history + pred_steps - 1]
        batch_size = self.batchsize
        nodes = self.nodes
        history = self.history
        out_channel = self.out_channel

        pred_all = []
        inputs = inputs.reshape(nodes, batch_size, -1) #[nodes, batch_size, window_size]

        if self.is_emb_conv:
            for step in range(pred_steps):
                # Embedding raw node features
                if step < history:
                    if reconstruction:
                        ins = inputs[:, :, step:history+step]
                    else:
                        ins = inputs[:, :, step:history] #[nodes, batch_size, history]
                if step == 0:
                    ins = ins.reshape(-1, 1, history)
                    h_emb = self.f_emb(ins)
                    h_emb = h_emb.reshape(nodes, -1, out_channel)
                    pred, hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                  h_emb=h_emb, hidden=h_emb, lam=lam_t)
                    y_hat = torch.poisson(pred)                              
                elif step < history:
                    if reconstruction:
                        h_emb = self.f_emb(ins.reshape(-1, 1, history))
                    else:
                        ins_cat = torch.cat((ins, y_hat), dim=-1)
                        ins_cat = ins_cat.reshape(-1, 1, history)
                        h_emb = self.f_emb(ins_cat)
                    h_emb = h_emb.reshape(nodes, -1, out_channel)
                    pred, hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                  h_emb=h_emb, hidden=hidden, lam=pred)
                    if not(reconstruction):
                        y_hat = torch.cat((y_hat, torch.poisson(pred)), dim=-1)
                else:
                    ins_cat = y_hat.reshape(-1, 1, history)
                    h_emb = self.f_emb(ins_cat)
                    h_emb = h_emb.reshape(nodes, -1, out_channel)
                    pred, hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                  h_emb=h_emb, hidden=hidden, lam=pred)
                    y_hat = torch.cat((y_hat[:, :, 1:], torch.poisson(pred)), dim=-1)
                pred_all.append(pred)
        
        else:
            for step in range(pred_steps):
                # Embedding raw node features
                if step < history:
                    if reconstruction:
                        ins = inputs[:, :, step:history+step]
                    else:
                        ins = inputs[:, :, step:history]
                if step == 0:
                    h_emb = self.f_emb(ins)
                    pred, hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                  h_emb=h_emb, hidden=h_emb, lam=lam_t)
                    y_hat = torch.poisson(pred)                              
                elif step < history:
                    if reconstruction:
                        h_emb = self.f_emb(ins)
                    else:
                        ins_cat = torch.cat((ins, y_hat), dim=-1)
                        h_emb = self.f_emb(ins_cat)
                    pred, hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                  h_emb=h_emb, hidden=hidden, lam=pred)
                    if not(reconstruction):
                        y_hat = torch.cat((y_hat, torch.poisson(pred)), dim=-1)
                else:
                    ins_cat = y_hat
                    h_emb = self.f_emb(ins_cat)
                    pred, hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                  h_emb=h_emb, hidden=hidden, lam=pred)
                    y_hat = torch.cat((y_hat[:, :, 1:], torch.poisson(pred)), dim=-1)
                pred_all.append(pred)
        preds = torch.stack(pred_all, dim=2)
        return preds

    def message(self, x_i, x_j, z):
        # Get edge features
        x_i, x_j = (x_i.reshape(z.shape[0], -1, self.hidden_dim), 
                    x_j.reshape(z.shape[0], -1, self.hidden_dim))
        x_edge = torch.cat((x_i, x_j), dim=-1)
        # [edges, bs, hdim]
        msg = torch.tanh(self.fc1_hat_e(x_edge)) # Why tanh in NRI?
        msg = F.dropout(msg, p=self.do_prob)
        msg = torch.tanh(self.fc2_hat_e(msg))
        msg = msg * z.reshape(-1, 1, 1)
        msg = msg.reshape(-1, self.batchsize * self.hidden_dim)
        return msg

    def update(self, aggr_msg, h_emb, hidden, lam):
        # The input to the message passing operation is the recurrent hidden state at the previous time step.
        # Aggregate MSGs
        aggr_msg = aggr_msg.reshape(self.nodes, -1, self.hidden_dim)
        node_feature = self.g1(h_emb)
        ins = torch.cat((aggr_msg, node_feature), dim=-1)
        ins_dim = ins.shape[-1]

        hidden = self.gru_cell(ins.reshape(-1, ins_dim), hidden.reshape(-1, self.hidden_dim))
        hidden = hidden.reshape(self.nodes, -1, self.hidden_dim)
        hidden = F.dropout(F.relu(self.fc1_out(hidden)), p=self.do_prob)
        pred = F.relu(lam + F.dropout(self.fc2_out(hidden), p=self.do_prob))
        return pred, hidden

class RNN_DEC_log(MessagePassing):
    def __init__(self, nodes, history, n_hid, batch_size, f_emb, g1, g1_out, act, do_prob=0.):
        super(RNN_DEC_log, self).__init__(aggr='add') # Eq 14
        self.hidden_dim = n_hid
        self.out_channel = n_hid
        self.batchsize = batch_size
        self.do_prob = do_prob
        self.nodes = nodes
        self.edges = nodes * (nodes-1)
        self.history = history
        
        self.f_emb, self.is_emb_conv = self.define_f_emb(n_in=history, n_hid=n_hid, function=f_emb, activation=act)
        self.g1, g1_dim = self.define_g1(g1_in=n_hid, g1_out=g1_out, function=g1, activation=act)
        self.gru_cell = nn.GRUCell(g1_dim+n_hid, n_hid)
        
        self.fc1_hat_e = nn.Linear(2 * n_hid, n_hid) # fc1 to fe
        self.fc2_hat_e = nn.Linear(n_hid, n_hid)

        self.fc1_out = nn.Linear(n_hid, n_hid)
        self.fc2_out = nn.Linear(n_hid, 1)

    def define_f_emb(self, n_in, n_hid, function, activation):
        f_emb = nn.Sequential()

        if function == 'conv':
            f_emb.add_module('f_emb', nn.Conv1d(1, n_hid, n_in))
            is_emb_conv = True
        elif function == 'mlp':
            f_emb.add_module('f_emb', nn.Linear(n_in, n_hid))
            is_emb_conv = False
        else:
            assert False, 'Invalid emb function type, try {mlp or conv}'

        if activation == 'sigmoid':
            f_emb.add_module('f_emb_act', nn.Sigmoid())
        elif activation == 'tanh':
            f_emb.add_module('f_emb_act', nn.Tanh())
        elif activation == 'relu':
            f_emb.add_module('f_emb_act', nn.ReLU())
        else:
            assert False, 'Invalid activation function type, try {sigmoid, tanh or relu}'
        
        return f_emb, is_emb_conv

    def define_g1(self, g1_in, g1_out, function, activation):
        g1 = nn.Sequential()

        if function == 'identity':
            g1_out = g1_in
            g1.add_module('g1', nn.Identity())
        elif function == 'mlp':
            g1.add_module('g1', nn.Linear(g1_in, g1_out))
        else:
            assert False, 'Invalid g1 function type, try {identity, mlp or }'

        if function != 'identity':
            if activation == 'sigmoid':
                g1.add_module('g1_act', nn.Sigmoid())
            elif activation == 'tanh':
                g1.add_module('g1_act', nn.Tanh())
            elif activation == 'relu':
                g1.add_module('g1_act', nn.ReLU())
            else:
                assert False, 'Invalid activation function type, try {sigmoid, tanh or relu}'
        
        return g1, g1_out
        
    def forward(self, inputs, edge_index, z, reconstruction=False, pred_steps=1, msg_hop=1):
        # [batch_size, num_neurons, time_steps]
        # num_timesteps should be [window_size = history + pred_steps - 1]
        nodes = self.nodes
        history = self.history
        out_channel = self.out_channel

        pred_all = []
        inputs = inputs.transpose(0, 1) # change dim to [num_neurons, batch_size, time_steps]

        if self.is_emb_conv:
            for step in range(pred_steps):
                # Embedding raw node features
                if step < history:
                    if reconstruction:
                        ins = inputs[:, :, step:history+step] # Predict only the next step
                    else:    
                        ins = inputs[:, :, step:history]
                if step == 0:
                    # Embed Input data X
                    ins = ins.reshape(-1, 1, history)
                    h_emb = self.f_emb(ins)
                    h_emb = h_emb.reshape(nodes, -1, out_channel)
                    # Progate MSGs for msg_hop
                    for k in range(msg_hop):
                        if k == 0:
                            hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                    h_emb=h_emb, hidden=h_emb)
                        else:
                            hidden = self.propagate(edge_index, x=hidden.reshape(nodes, -1), z=z, 
                                                    h_emb=hidden, hidden=hidden)
                    # Predict next step log Poisson Rate
                    pred = F.dropout(F.relu(self.fc1_out(hidden)), p=self.do_prob)
                    pred = F.dropout(self.fc2_out(hidden), p=self.do_prob)
                    y_hat = torch.poisson(pred.exp())                              
                elif step < history:
                    if reconstruction:
                        h_emb = self.f_emb(ins.reshape(-1, 1, history))
                    else:
                        ins_cat = torch.cat((ins, y_hat), dim=-1)
                        ins_cat = ins_cat.reshape(-1, 1, history)
                        h_emb = self.f_emb(ins_cat)
                    h_emb = h_emb.reshape(nodes, -1, out_channel)
                    # Progate MSGs for msg_hop
                    for k in range(msg_hop):
                        if k == 0:
                            hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                    h_emb=h_emb, hidden=h_emb)
                        else:
                            hidden = self.propagate(edge_index, x=hidden.reshape(nodes, -1), z=z, 
                                                    h_emb=hidden, hidden=hidden)
                    # Predict next step log Poisson Rate
                    pred = F.dropout(F.relu(self.fc1_out(hidden)), p=self.do_prob)
                    pred = F.dropout(self.fc2_out(hidden), p=self.do_prob)
                    # Next-step input X
                    if not(reconstruction):
                        y_hat = torch.cat((y_hat, torch.poisson(pred.exp())), dim=-1)
                else:
                    if step % 10000 == 0:
                        print(step)
                    ins_cat = y_hat.reshape(-1, 1, history)
                    h_emb = self.f_emb(ins_cat)
                    h_emb = h_emb.reshape(nodes, -1, out_channel)
                    # Progate MSGs for msg_hop
                    for k in range(msg_hop):
                        if k == 0:
                            hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                    h_emb=h_emb, hidden=h_emb)
                        else:
                            hidden = self.propagate(edge_index, x=hidden.reshape(nodes, -1), z=z, 
                                                    h_emb=hidden, hidden=hidden)
                    # Predict next step log Poisson Rate
                    pred = F.dropout(F.relu(self.fc1_out(hidden)), p=self.do_prob)
                    pred = F.dropout(self.fc2_out(hidden), p=self.do_prob)
                    # Next-step input X
                    y_hat = torch.cat((y_hat[:, :, 1:], torch.poisson(pred.exp())), dim=-1)
                pred_all.append(pred)
        
        else:
            for step in range(pred_steps):
                # Embedding raw node features
                if step < history:
                    if reconstruction:
                        ins = inputs[:, :, step:step+history]
                    else:
                        ins = inputs[:, :, step:history]
                if step == 0:
                    h_emb = self.f_emb(ins)
                    # Progate MSGs for msg_hop
                    for k in range(msg_hop):
                        if k == 0:
                            hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                    h_emb=h_emb, hidden=h_emb)
                        else:
                            hidden = self.propagate(edge_index, x=hidden.reshape(nodes, -1), z=z, 
                                                    h_emb=hidden, hidden=hidden)
                    # Predict next step log Poisson Rate
                    pred = F.dropout(F.relu(self.fc1_out(hidden)), p=self.do_prob)
                    pred = F.dropout(self.fc2_out(hidden), p=self.do_prob)
                    # Next-step input X
                    y_hat = torch.poisson(pred.exp())                              
                elif step < history:
                    if reconstruction:
                        h_emb = self.f_emb(ins)
                    else:
                        ins_cat = torch.cat((ins, y_hat), dim=-1)
                        h_emb = self.f_emb(ins_cat)
                    # Progate MSGs for msg_hop
                    for k in range(msg_hop):
                        if k == 0:
                            hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                    h_emb=h_emb, hidden=h_emb)
                        else:
                            hidden = self.propagate(edge_index, x=hidden.reshape(nodes, -1), z=z, 
                                                    h_emb=hidden, hidden=hidden)
                    # Predict next step log Poisson Rate
                    pred = F.dropout(F.relu(self.fc1_out(hidden)), p=self.do_prob)
                    pred = F.dropout(self.fc2_out(hidden), p=self.do_prob)
                    # Next-step input X
                    if not(reconstruction):
                        y_hat = torch.cat((y_hat, torch.poisson(pred.exp())), dim=-1)
                else:
                    if step % 10000 == 0:
                        print(step)
                    ins_cat = y_hat
                    h_emb = self.f_emb(ins_cat)
                    # Progate MSGs for msg_hop
                    for k in range(msg_hop):
                        if k == 0:
                            hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                    h_emb=h_emb, hidden=h_emb)
                        else:
                            hidden = self.propagate(edge_index, x=hidden.reshape(nodes, -1), z=z, 
                                                    h_emb=hidden, hidden=hidden)
                    # Predict next step log Poisson Rate
                    pred = F.dropout(F.relu(self.fc1_out(hidden)), p=self.do_prob)
                    pred = F.dropout(self.fc2_out(hidden), p=self.do_prob)
                    # Next-step input X
                    y_hat = torch.cat((y_hat[:, :, 1:], torch.poisson(pred.exp())), dim=-1)
                pred_all.append(pred)
        preds = torch.stack(pred_all, dim=2)
        return preds

    def message(self, x_i, x_j, z):
        # Get edge features
        x_i, x_j = (x_i.reshape(z.shape[0], -1, self.hidden_dim), 
                    x_j.reshape(z.shape[0], -1, self.hidden_dim))
        x_edge = torch.cat((x_i, x_j), dim=-1)
        # [edges, bs, hdim]
        msg = torch.tanh(self.fc1_hat_e(x_edge)) # Why tanh in NRI?
        msg = F.dropout(msg, p=self.do_prob)
        msg = torch.tanh(self.fc2_hat_e(msg))
        msg = msg * z.reshape(-1, 1, 1)
        msg = msg.reshape(z.shape[0], -1)
        return msg

    def update(self, aggr_msg, h_emb, hidden):
        # The input to the message passing operation is the recurrent hidden state at the previous time step.
        # Aggregate MSGs
        aggr_msg = aggr_msg.reshape(self.nodes, -1, self.hidden_dim)
        node_feature = self.g1(h_emb)
        ins = torch.cat((aggr_msg, node_feature), dim=-1)
        ins_dim = ins.shape[-1]

        hidden = self.gru_cell(ins.reshape(-1, ins_dim), hidden.reshape(-1, self.hidden_dim))
        hidden = hidden.reshape(self.nodes, -1, self.hidden_dim)
        return hidden

class GNN_ENC_mousehd(MessagePassing):
    def __init__(self, nodes, history, n_hid, f_emb, act, do_prob=0.):
        super(GNN_ENC_mousehd, self).__init__(aggr='add') # Eq 14
        self.hidden_dim = n_hid
        self.out_channel = n_hid
        self.do_prob = do_prob
        self.nodes = nodes
        self.edges = nodes * (nodes-1)
        self.history = history
        
        self.f_emb_spk, self.is_emb_conv = self.define_f_emb(n_in=history, n_hid=n_hid, function=f_emb, activation=act)
        self.f_emb_ang, _ = self.define_f_emb(n_in=history, n_hid=n_hid, function=f_emb, activation=act)
        self.gru_cell = nn.GRUCell(3 * n_hid, n_hid)
        
        self.fc1_hat_e = nn.Linear(2 * n_hid, n_hid) # fc1 to fe
        self.fc2_hat_e = nn.Linear(n_hid, n_hid)

        self.fc1_out_poiss = nn.Linear(n_hid, n_hid)
        self.fc1_out_ang = nn.Linear(n_hid, n_hid)
        self.fc2_out_poiss = nn.Linear(n_hid, 1)
        self.fc2_out_ang = nn.Linear(n_hid, 1)


    def define_f_emb(self, n_in, n_hid, function, activation):
        f_emb = nn.Sequential()

        if function == 'conv':
            f_emb.add_module('f_emb', nn.Conv1d(1, n_hid, n_in))
            is_emb_conv = True
        elif function == 'mlp':
            f_emb.add_module('f_emb', nn.Linear(n_in, n_hid))
            is_emb_conv = False
        else:
            assert False, 'Invalid emb function type, try {mlp or conv}'

        if activation == 'sigmoid':
            f_emb.add_module('f_emb_act', nn.Sigmoid())
        elif activation == 'tanh':
            f_emb.add_module('f_emb_act', nn.Tanh())
        elif activation == 'relu':
            f_emb.add_module('f_emb_act', nn.ReLU())
        else:
            assert False, 'Invalid activation function type, try {sigmoid, tanh or relu}'
        
        return f_emb, is_emb_conv
        
    def forward(self, inputs, ang_window, edge_index, z, pref_hd, reconstruction=False, pred_steps=1, msg_hop=1):
        # [batch_size, num_neurons, time_steps]
        # num_timesteps should be [window_size = history + pred_steps - 1]
        nodes = self.nodes
        history = self.history
        out_channel = self.out_channel

        pred_all, ang_all = [], []
        inputs = inputs.transpose(0, 1) # change dim to [num_neurons, batch_size, time_steps]

        for step in range(pred_steps):
            # Embedding raw node features
            if step < history:
                ins = inputs[:, :, step:history+step] if reconstruction else inputs[:, :, step:history]
                ang = ang_window[:, step:history+step] if reconstruction else ang_window[:, step:history]

                if self.is_emb_conv:
                    ins = ins.reshape(-1, 1, history)
                    ang = ang.reshape(-1, 1, history)

                if step == 0:
                    h_emb = self.f_emb_spk(ins)
                    ang_emb = self.f_emb_ang(ang)
                
                    if self.is_emb_conv:
                        h_emb = h_emb.reshape(nodes, -1, out_channel) #[nodes, batch, hid]
                        ang_emb = ang_emb.reshape(-1, out_channel) #[batch, hid]

                elif step < history:
                    if reconstruction:
                        h_emb = self.f_emb_spk(ins)
                        ang_emb = self.f_emb_ang(ang)
                
                        if self.is_emb_conv:
                            h_emb = h_emb.reshape(nodes, -1, out_channel)
                            ang_emb = ang_emb.reshape(-1, out_channel)

                    else:
                        ins_cat = torch.cat((ins, y_hat), dim=-1)
                        ang_cat = torch.cat((ang, ang_hat), dim=-1)
                        
                        if self.is_emb_conv:
                            ins_cat = ins_cat.reshape(-1, 1, history)
                            ang_cat = ang_cat.reshape(-1, 1, history)

                        h_emb = self.f_emb_spk(ins_cat)
                        ang_emb = self.f_emb_ang(ang_cat)

                        if self.is_emb_conv:
                            h_emb = h_emb.reshape(nodes, -1, out_channel)
                            ang_emb = ang_emb.reshape(-1, out_channel)
                    
                else:
                    if step % 10000 == 0:
                        print(step)
                    
                    ins_cat = y_hat.reshape(-1, 1, history) if self.is_emb_conv else y_hat
                    ang_cat = ang_hat.reshape(-1, 1, history) if self.is_emb_conv else ang_hat

                    h_emb = self.f_emb_spk(ins_cat)
                    ang_emb = self.f_emb_ang(ang_cat)

                    if self.is_emb_conv:
                        h_emb = h_emb.reshape(nodes, -1, out_channel)
                        ang_emb = ang_emb.reshape(-1, out_channel)

                # Progate MSGs for msg_hop
                for k in range(msg_hop):
                    if k == 0:
                        hidden = self.propagate(edge_index, x=h_emb.reshape(nodes, -1), z=z, 
                                                h_emb=h_emb, ang_emb=ang_emb, hidden=h_emb)
                    else:
                        hidden = self.propagate(edge_index, x=hidden.reshape(nodes, -1), z=z, 
                                                h_emb=hidden, ang_emb=ang_emb, hidden=hidden)

                # Predict next step log Poisson Rate
                pred = F.relu(self.fc1_out_poiss(hidden))
                pred = self.fc2_out_poiss(hidden)

                ang_weight = F.relu(self.fc1_out_ang(hidden))
                ang_weight = self.fc2_out_ang(ang_weight) # [Neurons, Batch, 1]
                
                # Pooling with feated Pref HD
                ang_next = ang_weight * pref_hd.reshape(nodes, 1, 1) # [Neurons, Batch, 1] [Neurons, 1, 1]
                ang_next = ang_next.sum(dim=0) # [Batch, 1]

                if step == 0:
                    y_hat = torch.poisson(pred.exp())
                    ang_hat = ang_next    

                if not reconstruction and step > 0:
                    y_hat = torch.cat((y_hat, torch.poisson(pred.exp())), dim=-1)
                    ang_hat = torch.cat((ang_hat, ang_next), dim=-1)

            pred_all.append(pred)
            ang_all.append(ang_next)

        preds = torch.stack(pred_all, dim=2) # [Neurons, Batch, Pred_T, 1]
        angs = torch.stack(ang_all, dim=1) # [Batch, Pred_T, 1]
        return preds, angs

    def message(self, x_i, x_j, z):
        # Get edge features
        x_i, x_j = (x_i.reshape(z.shape[0], -1, self.hidden_dim), 
                    x_j.reshape(z.shape[0], -1, self.hidden_dim))
        x_edge = torch.cat((x_i, x_j), dim=-1)
        # [edges, bs, hdim]
        msg = torch.tanh(self.fc1_hat_e(x_edge)) # Why tanh in NRI?
        msg = F.dropout(msg, p=self.do_prob)
        msg = torch.tanh(self.fc2_hat_e(msg))
        msg = msg * z.reshape(-1, 1, 1)
        msg = msg.reshape(z.shape[0], -1)
        return msg

    def update(self, aggr_msg, h_emb, ang_emb, hidden):
        # The input to the message passing operation is the recurrent hidden state at the previous time step.
        # Aggregate MSGs, expand angular embedding to all neurons
        aggr_msg = aggr_msg.reshape(self.nodes, -1, self.hidden_dim)
        ang_emb = ang_emb.unsqueeze(0).expand_as(aggr_msg)
        ins = torch.cat((aggr_msg, h_emb, ang_emb), dim=-1)
        ins_dim = ins.shape[-1]

        hidden = self.gru_cell(ins.reshape(-1, ins_dim), hidden.reshape(-1, self.hidden_dim))
        hidden = hidden.reshape(self.nodes, -1, self.hidden_dim)
        return hidden