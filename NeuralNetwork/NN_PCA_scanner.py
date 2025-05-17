import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import NN_Library as NN
import time as clock
import os

from matplotlib import cm, colors
from matplotlib.widgets import Slider


if __name__ == '__main__':

    start_cpu = clock.process_time()

    script_directory = os.path.dirname(os.path.abspath(__file__))
    mother_directory = os.path.dirname(script_directory)
    dataset_path = os.path.join(mother_directory, 'Dataset')

    pc_mat = pd.read_csv(dataset_path+'/eigenvectors.csv', sep=',', index_col=0)
    pc_mat = pc_mat.to_numpy()

    patterns = pd.read_csv(dataset_path+'/iono_trainPatt.csv', header=0, index_col=0)
    patterns = patterns.to_numpy()
    patterns = patterns.dot(pc_mat)
    patterns = NN.sigmoid(patterns) #normalize the data with a sigmoid function
    labels = pd.read_csv(dataset_path+'/iono_trainLab.csv', header=0, index_col=0)
    labels = np.ravel(labels.to_numpy())
    label0 = 'b'
    label1 = 'g'

    good = 0
    bad = 0

    for l in labels:
        if l == label1:
            good += 1
        if l == label0:
            bad += 1

    print('Good signals:', good)
    print('Bad signals:', bad)
    
    epoch_step = 1
    epoch_range = range(1, 101, epoch_step)
    k_range = range(2, 11, 1)
    lr_range = np.arange(1e-4, 1e-3, 1e-4)

    res = NN.NN_binary_scanner(epoch_range, k_range, lr_range, patterns, labels, label0, label1)
    
    if not os.path.exists(script_directory+f'/NN PCA'):
        os.makedirs(script_directory+f'/NN PCA')
    
    if not os.path.exists(script_directory+f'/NN PCA/score_heatmaps'):
        os.makedirs(script_directory+f'/NN PCA/score_heatmaps')
    
    if not os.path.exists(script_directory+f'/NN PCA/loss_heatmaps'):
        os.makedirs(script_directory+f'/NN PCA/loss_heatmaps')
    
    sens = res['Sens List']
    spec = res['Spec List']
    loss = res['Loss List']
    scores = np.concatenate((sens, spec), axis = 0)
    normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))
    loss_norm = colors.Normalize(vmin=np.min(loss), vmax=np.max(loss))

    for i,k in enumerate(k_range):
        fig, ax = plt.subplots(1,2, figsize=(16,9))

        sens_colormesh = NN.heatmap_plotter(ax[0], lr_range, epoch_range, sens[:,i,:], '{:.2f}', "Sensitivity", 'Learning Rate', 'Number of epochs', normalization, cm.viridis)
        spec_colormesh = NN.heatmap_plotter(ax[1], lr_range, epoch_range, spec[:,i,:], '{:.2f}', "Specificity", 'Learning Rate', 'Number of epochs', normalization, cm.viridis)
        
        fig.suptitle(f'{k}-Fold')
        fig.colorbar(spec_colormesh,  orientation='vertical')
        fig.savefig(script_directory+f'/NN PCA/score_heatmaps/heatmap_{k}_fold.png', dpi=240)
        plt.close(fig)
                
        fig_loss, ax_loss = plt.subplots( figsize=(16,9))

        loss_colormesh = NN.heatmap_plotter(ax_loss, lr_range, epoch_range, loss[:,i,:], '{:.1e}', 'Loss', 'Learning Rate', 'Number of epochs', loss_norm, cm.viridis )

        fig_loss.suptitle(f'{k}-Fold')
        fig_loss.colorbar(loss_colormesh,  orientation='vertical')
        fig_loss.savefig(script_directory+f'/NN PCA/loss_heatmaps/heatmap_{k}_fold.png', dpi=240)
        plt.close(fig_loss)

    print("CPU Time:" + str(round((clock.process_time() - start_cpu)/60)) + "'" + str(round((clock.process_time() - start_cpu)%60)) + "''")
    # plt.show()

    fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=(16,9))
    # fig1.subplots_adjust(0.2, 0.2)
    ax1[0].scatter(lr_range, sens[0,0,:], color='C1')
    ax1[0].errorbar(lr_range, sens[0,0,:], yerr=np.std(sens[0,0,:]), color='C1')
    ax1[0].grid(True)
    ax1[0].set_ylim(0, 1.1)
    ax1[0].set_title('Sensitivity')

    ax1[1].scatter(lr_range, spec[0,0,:], color='C2')
    ax1[1].errorbar(lr_range, spec[0,0,:], yerr=np.std(spec[0,0,:]), color='C2')
    ax1[1].grid(True)
    ax1[1].set_ylim(0, 1.1)
    ax1[1].set_title('Specificity')

    ax1[1].set_xlabel('Learning rate')
    fig1.suptitle(f'Metrics: {k_range[0]}-folds, {epoch_range[0]} epochs')

    axSlide_k1 = fig1.add_axes([0.05, 0.05, 0.05, 0.4])
    kSlide1 = Slider(ax = axSlide_k1, label = "Number of folds", valmin=k_range[0], valmax=k_range[-1], valstep = 1, valinit=k_range[0], orientation='vertical')
    axSlide_ep = fig1.add_axes([0.05, 0.55, 0.05, 0.4])
    epSlide = Slider(ax = axSlide_ep, label = "Number of epochs", valmin=epoch_range[0], valmax=epoch_range[-1], valstep = epoch_step, valinit=epoch_range[0], orientation='vertical')

def update_metrics_1(val):
    i_k = k_range.index(kSlide1.val)
    i_ep = epoch_range.index(epSlide.val)

    ax1[0].clear()
    ax1[0].scatter(lr_range, sens[i_ep,i_k,:], color='C1')
    ax1[0].errorbar(lr_range, sens[i_ep,i_k,:], yerr=np.std(sens[i_ep,i_k,:]), color='C1')
    ax1[0].grid(True)
    ax1[0].set_ylim(0, 1.1)
    ax1[0].set_title('Sensitivity')

    ax1[1].clear()
    ax1[1].scatter(lr_range, spec[i_ep,i_k,:], color='C2')
    ax1[1].errorbar(lr_range, spec[i_ep,i_k,:], yerr=np.std(spec[i_ep,i_k,:]), color='C2')
    ax1[1].grid(True)
    ax1[1].set_ylim(0, 1.1)
    ax1[1].set_title('Specificity')

    ax1[1].set_xlabel('Learning rate')
    fig1.suptitle(f'Metrics: {k_range[i_k]}-folds, {epoch_range[i_ep]} epochs')

    fig1.canvas.draw_idle()

kSlide1.on_changed(update_metrics_1)
epSlide.on_changed(update_metrics_1)

fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(16,9))
# fig2.subplots_adjust(0.2, 0.2)
ax2[0].scatter(epoch_range, sens[:,0,0], color='C1')
ax2[0].errorbar(epoch_range, sens[:,0,0], yerr=np.std(sens[:,0,0]), color='C1')
ax2[0].grid(True)
ax2[0].set_ylim(0, 1.1)
ax2[0].set_title('Sensitivity')

ax2[1].scatter(epoch_range, spec[:,0,0], color='C2')
ax2[1].errorbar(epoch_range, spec[:,0,0], yerr=np.std(sens[:,0,0]), color='C2')
ax2[1].grid(True)
ax2[1].set_ylim(0, 1.1)
ax2[1].set_title('Specificity')

ax2[1].set_xlabel('Epochs')
fig2.suptitle(f'Metrics: {k_range[0]}-folds, {lr_range[0]:0.5f} LR')

axSlide_k2 = fig2.add_axes([0.05, 0.05, 0.05, 0.4])
kSlide2 = Slider(ax = axSlide_k2, label = "Number of folds", valmin=k_range[0], valmax=k_range[-1], valstep = 1, valinit=k_range[0], orientation='vertical')
axSlide_lr = fig2.add_axes([0.05, 0.55, 0.05, 0.4])
lrSlide = Slider(ax = axSlide_lr, label = "Learning Rate", valmin=0, valmax=len(lr_range)-1, valstep = 1, valinit=0, orientation='vertical')

def update_metrics_2(val):
    i_k = k_range.index(kSlide2.val)
    i_lr = int(lrSlide.val)

    ax2[0].clear()
    ax2[0].scatter(epoch_range, sens[:,i_k,i_lr], color='C1')
    ax2[0].errorbar(epoch_range, sens[:,i_k,i_lr], yerr=np.std(sens[:,i_k,i_lr]), color='C1')
    ax2[0].grid(True)
    ax2[0].set_ylim(0, 1.1)
    ax2[0].set_title('Sensitivity')

    ax2[1].clear()
    ax2[1].scatter(epoch_range, spec[:,i_k,i_lr], color='C2')
    ax2[1].errorbar(epoch_range, spec[:,i_k,i_lr], yerr=np.std(spec[:,i_k,i_lr]), color='C2')
    ax2[1].grid(True)
    ax2[1].set_ylim(0, 1.1)
    ax2[1].set_title('Specificity')

    ax2[1].set_xlabel('Epochs')
    fig2.suptitle(f'Metrics: {k_range[i_k]}-folds, {lr_range[i_lr]:0.1e} LR')

    fig2.canvas.draw_idle()

kSlide2.on_changed(update_metrics_2)
lrSlide.on_changed(update_metrics_2)

fig3, ax3 = plt.subplots(2, 1, figsize=(16,9))
fig3.subplots_adjust(hspace=0.5)
ax3[0].plot(epoch_range, loss[:,0,0], color='C1')
# ax3[0].errorbar(epoch_range, loss[:,0,0], yerr=np.std(loss[:,0,0]), color='C1')
ax3[0].grid(True)
ax3[0].set_ylim(np.min(loss[:,0,0]) - np.min(loss[:,0,0])/5, np.max(loss[:,0,0]) + np.max(loss[:,0,0])/5)
ax3[0].set_title(f'Loss at {lr_range[0]:0.1e} LR')
ax3[0].set_xlabel('Epochs')

ax3[1].plot(lr_range, loss[0,0,:], color='C2')
# ax3[1].errorbar(lr_range, loss[i_ep,i_k,:], yerr=np.std(loss[i_ep,i_k,:]), color='C2')
ax3[1].grid(True)
ax3[1].set_ylim(np.min(loss[0,0,:]) - np.min(loss[0,0,:])/5 , np.max(loss[0,0,:]) + np.max(loss[0,0,:])/5)
ax3[1].set_title(f'Loss at {epoch_range[0]} epochs')
ax3[1].set_xlabel('Learning Range')

fig3.suptitle(f'Loss in Training: {k_range[0]} Folds')

axSlide_k3 = fig3.add_axes([0.93, 0.05, 0.05, 0.4])
kSlide3 = Slider(ax = axSlide_k3, label = "Number of folds", valmin=k_range[0], valmax=k_range[-1], valstep = 1, valinit=k_range[0], orientation='vertical')
axSlide_lr3 = fig3.add_axes([0.04, 0.05, 0.05, 0.4])
lrSlide3 = Slider(ax = axSlide_lr3, label = "Learning Rate", valmin=0, valmax=len(lr_range)-1, valstep = 1, valinit=0, orientation='vertical')
axSlide_ep3 = fig3.add_axes([0.04, 0.55, 0.05, 0.4])
epSlide3 = Slider(ax = axSlide_ep3, label = "Number of epochs", valmin=epoch_range[0], valmax=epoch_range[-1], valstep = epoch_step, valinit=epoch_range[0], orientation='vertical')

def update_loss(val):
    i_k = k_range.index(kSlide3.val)
    i_lr = int(lrSlide3.val)
    i_ep = epoch_range.index(epSlide3.val)

    ax3[0].clear()
    ax3[0].plot(epoch_range, loss[:,i_k,i_lr], color='C1')
    # ax3[0].errorbar(epoch_range, loss[:,i_k,i_lr], yerr=np.std(loss[:,i_k,i_lr]), color='C1')
    ax3[0].grid(True)
    ax3[0].set_ylim(np.min(loss[:,i_k,i_lr]) - np.min(loss[:,i_k,i_lr])/5, np.max(loss[:,i_k,i_lr]) + np.max(loss[:,i_k,i_lr])/5)
    ax3[0].set_title(f'Loss at {lr_range[i_lr]:0.1e} LR')
    ax3[0].set_xlabel('Epochs')

    ax3[1].clear()
    ax3[1].plot(lr_range, loss[i_ep,i_k,:], color='C2')
    # ax3[1].errorbar(lr_range, loss[i_ep,i_k,:], yerr=np.std(loss[i_ep,i_k,:]), color='C2')
    ax3[1].grid(True)
    ax3[1].set_ylim(np.min(loss[i_ep,i_k,:]) - np.min(loss[i_ep,i_k,:])/5 , np.max(loss[i_ep,i_k,:]) + np.max(loss[i_ep,i_k,:])/5)
    ax3[1].set_title(f'Loss at {epoch_range[i_ep]} epochs')
    ax3[1].set_xlabel('Learning Range')

    fig3.suptitle(f'Loss in Training: {k_range[i_k]} Folds')

    fig3.canvas.draw_idle()

kSlide3.on_changed(update_loss)
lrSlide3.on_changed(update_loss)
epSlide3.on_changed(update_loss)

plt.show()