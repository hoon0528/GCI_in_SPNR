import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import z_to_adj_plot, scale_w_hat_scipy

def plot_adj(test, nodes, plot_folder, suffix=''):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    adj_z = np.zeros(nodes * nodes)

    for i in range(nodes - 1):
        adj_z[i*(nodes+1) + 1:(i+1)*(nodes+1)] = test[i*nodes: (i+1)*nodes]
        
    adj_z = adj_z.reshape(nodes, nodes)
    div_0 = make_axes_locatable(axes)
    cax_0 = div_0.append_axes("right", size="5%", pad=0.05)
    cax_0.tick_params(axis='both', labelsize=24)

    fig.colorbar(axes.imshow(adj_z, cmap='hot'), cax=cax_0)
    
    axes.tick_params(axis='both', labelsize=24)
    plt.savefig(plot_folder + '/adj{}.svg'.format(suffix), bbox_inches='tight')
    plt.close('all')

def plot_w_single(w_hat, tar, w_hat_std, nodes, plot_folder, suffix='', gif_mode=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    idx_num = int(nodes/2)

    w_hat_plot = np.roll(w_hat, idx_num)
    w_tar_plot = np.roll(tar, idx_num)
    w_hat_plot[idx_num] = None
    w_tar_plot[idx_num] = None
    x = np.arange(nodes)

    axes.plot(w_hat_plot, color='royalblue', label='W_hat(single neuron)', alpha=0.7)
    axes.plot(w_tar_plot, color='darkorange', label='Target')
    axes.fill_between(x, w_hat_plot - w_hat_std, w_hat_plot + w_hat_std,
                      color='royalblue', label='1 sigma range', alpha=0.2)

    axes.tick_params(axis='x', labelsize=10)
    axes.tick_params(axis='y', labelsize=10)

    axes.tick_params(axis='x', labelsize=10)
    axes.tick_params(axis='y', labelsize=10)

    plt.legend(loc=3, prop={'size': 8})
    if gif_mode:
        plt.savefig(plot_folder + '/ws{}.png'.format(suffix), bbox_inches='tight')
    else:
        plt.savefig(plot_folder + '/ws{}.svg'.format(suffix), bbox_inches='tight')
    plt.close('all')

def plot_lambda_raster(im_o, bin_size, nodes, plot_folder, suffix, if_plot=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    time = im_o.shape[0] / bin_size
    extent = [0, time, 0, nodes]
    h_over_w = 0.5
    aspect = h_over_w * (time/nodes)

    im_o = axes.imshow(np.transpose(im_o), cmap='binary', extent=extent)

    axes.set_aspect(aspect)
    axes.tick_params(axis='x', labelsize=10)
    axes.tick_params(axis='y', labelsize=10)

    cax_o = fig.add_axes([axes.get_position().x1+0.01,axes.get_position().y0,0.02,axes.get_position().height])

    cbar1 = plt.colorbar(im_o, cax=cax_o)
    cbar1.ax.tick_params(axis='y', labelsize=20)
    if if_plot:
        plt.show()
        plt.close('all')
    else:
        plt.savefig(plot_folder + '/lamall_{}.svg'.format(suffix), bbox_inches='tight')
        plt.close('all')

def plot_loss(losses, plot_folder, suffix, ylim=False):
    epochs, n_losses = losses.shape[0], losses.shape[1] # should be modified
    loss_plot = [losses[:,0], losses[:,1]]
    loss_sort = np.sort(losses, axis=None)
    lval_max = np.max(losses[-50:,1], axis=None)
    ymin = loss_sort[0] - (loss_sort[1] - loss_sort[0])
    ymax = np.abs((lval_max - loss_sort[0])) * 2 + loss_sort[0]
    #print(ymin, ymax)

    c_list = ['darkorange', 'royalblue']
    lable_list = ['Training', 'Validation']
    c_list, lable_list = c_list[:n_losses], lable_list[:n_losses]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    x = np.arange(epochs) + 1

    for i in range(len(c_list)):
        axes.plot(x, loss_plot[i], c=c_list[i], alpha=1, label=lable_list[i])

    axes.set_xlabel('Epochs', fontsize=30)
    if ylim:
        axes.set_ylim(ymin, ymax)
    axes.tick_params(axis='x', labelsize=20)
    axes.tick_params(axis='y', labelsize=20)
    plt.legend(prop={'size': 20})
    plt.savefig(plot_folder + '/loss_{}.svg'.format(suffix), bbox_inches='tight')    

def plot_single_neuron_lam(lam_tar, lam_pred_list, bin_size, time_steps, neuron_index, 
                           start_step, clist, llist, k, plot_folder, suffix=''):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    x_lam = (np.arange(time_steps) + 1 + start_step) / bin_size
    
    axes.plot(x_lam, lam_tar[start_step:start_step+time_steps, neuron_index]/k, label='Target', color='darkorange')
    for i in range(len(lam_pred_list)):
        axes.plot(x_lam, lam_pred_list[i][start_step:start_step+time_steps, neuron_index], 
                  label=llist[i], color=clist[i], alpha=0.5)

    axes.tick_params(axis='x', labelsize=20)
    axes.tick_params(axis='y', labelsize=20)

    plt.legend(loc=4, prop={'size': 16})
    plt.savefig(plot_folder + '/{}ls_n{}_{}.svg'.format(suffix, neuron_index, start_step), bbox_inches='tight')
    plt.close('all')

def get_lam_pred_list(exp_list):
    lam_pred_list = []

    for item in exp_list:
        lam_file = 'plot/exp{}/lam_pred.npy'.format(item)
        lam_pred_list.append(np.load(lam_file))

    return lam_pred_list

def make_gif_z(z_file, W, n_nodes, epochs, plot_folder):
    inf_err_list = []
    images = []
    fnames = []
    gif_imgs_folder = plot_folder + '/z_gif'

    if not os.path.exists(gif_imgs_folder):
        os.makedirs(gif_imgs_folder)

    for i in range(epochs):
        fnames.append(gif_imgs_folder + '/ws{}.png'.format(i))
        w1_hat = z_to_adj_plot(z_file[i, :, :].reshape(-1), n_nodes)
        z1_scaler, z1_inf_err, z1_hat_vec, z1_tar_vec, z1_hat_std = scale_w_hat_scipy(w1_hat, W)
        z1_plot = z1_scaler[0] * z1_hat_vec + z1_scaler[1]

        #plot_adj(z_file[i, :, :].reshape(-1), n_nodes, 'fig/wa_{}'.format(i))    
        plot_w_single(z1_plot, z1_tar_vec, z1_hat_std, n_nodes, gif_imgs_folder, suffix=i, gif_mode=True)
        inf_err_list.append(z1_inf_err.item())

    for fname in fnames:
        images.append(imageio.imread(fname))

    imageio.mimsave(plot_folder + '/gif_w.gif', images)
    return inf_err_list

def plot_corr(corr_mat, plot_folder, suffix):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    div_0 = make_axes_locatable(axes)
    cax_0 = div_0.append_axes("right", size="5%", pad=0.05)
    cax_0.tick_params(axis='both', labelsize=24)

    fig.colorbar(axes.imshow(corr_mat, cmap='hot'), cax=cax_0)
    axes.tick_params(axis='both', labelsize=24)
    plt.savefig(plot_folder + '/corr_{}'.format(suffix), bbox_inches='tight')
    plt.close('all')

def multi_z_plots(fig_x, fig_y, figsize, nodes, z_list, plot_folder, suffix, no_axis=False):
    fig, axes = plt.subplots(nrows=fig_x, ncols=fig_y, figsize=figsize)
    v_min, v_max = np.min(z_list[0]), np.max(z_list[0]) #Fix colorbar to Target W

    w_list = []
    for j in range(len(z_list)):
        adj_z = np.zeros(nodes * nodes)
        for i in range(nodes - 1):
            adj_z[i*(nodes+1) + 1:(i+1)*(nodes+1)] = z_list[j][i*nodes: (i+1)*nodes]
        adj_z = adj_z.reshape(nodes, nodes)
        w_list.append(adj_z)
    
    for idx, ax in enumerate(axes.flat):
        im = ax.imshow(w_list[idx], cmap='hot', vmin=v_min, vmax=v_max)

        if no_axis:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            if idx>0:
                ax.yaxis.set_visible(False)
            fig.colorbar(im, ax=axes.ravel().tolist())

    plt.savefig(plot_folder + '/adj{}.svg'.format(suffix), bbox_inches='tight')
    plt.close('all')

def multi_ws_plots(fig_x, fig_y, figsize, nodes, std_list, z_list, tar, plot_folder, 
                   suffix, no_axis=False, no_legend=False, if_ylim=False):
    fig, axes = plt.subplots(nrows=fig_x, ncols=fig_y, figsize=figsize)
    idx_num = int(nodes/2)
    y_max = tar.max() + 4e-4
    y_min = tar.min() - 1e-4

    ws_hat_list = []
    ws_tar = np.roll(tar, idx_num)
    ws_tar[idx_num] = None

    for z in z_list:
        ws_hat = np.roll(z, idx_num)
        ws_hat[idx_num] = None
        ws_hat_list.append(ws_hat)

    x = np.arange(nodes)
    for idx, ax in enumerate(axes.flat):
        if idx>=0:
            ax.plot(ws_hat_list[idx], color='royalblue', label=r'$\bar{\omega}$', alpha=1, linewidth=4)
            ax.fill_between(x, ws_hat_list[idx] - std_list[idx], ws_hat_list[idx] + std_list[idx],
                            color='royalblue', label=r'$1\sigma$ range', alpha=0.2)
            ax.plot(ws_tar, color='darkorange', label=r'$\omega$', linewidth=4, alpha=0.5)
        if not(no_legend):
            leg = ax.legend(loc=4, prop={'size': 36}, frameon=False)
            leg.get_frame().set_linewidth(0.0)

        if if_ylim:
            ax.set_ylim(y_min, y_max)

        if no_axis:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            if idx>0:
                ax.yaxis.set_visible(True)

    plt.savefig(plot_folder + '/ws{}.svg'.format(suffix), bbox_inches='tight')
    plt.close('all')

def plot_convfilt(filts, plot_folder, suffix, no_axis, y_margin):
    fig_x = filts.shape[0]
    fig, axes = plt.subplots(ncols=1, nrows=fig_x, figsize=(5, fig_x + y_margin))

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for idx, ax in enumerate(axes.flat):
        ax.plot(filts[idx, 0, :])
        #im = ax.plot(filts[idx, 0, :])
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(im, cax=cax, orientation='vertical')

        ax.xaxis.set_visible(False)
        
        if no_axis:
            ax.yaxis.set_visible(False)

    plt.savefig(plot_folder + '/conv{}.svg'.format(suffix), bbox_inches='tight')
    plt.close('all')

        
