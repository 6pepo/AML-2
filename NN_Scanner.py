import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import NN_Library as NN
import time as clock
import os

from matplotlib import cm, colors

if __name__ == '__main__':

    start_cpu = clock.process_time()

    patterns = pd.read_csv("iono_trainPatt.csv", header=0, index_col=0)
    patterns = patterns.to_numpy()
    patterns = NN.sigmoid(patterns) #normalize the data with a sigmoid function
    labels = pd.read_csv("iono_trainLab.csv", header=0, index_col=0)
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
    
    epoch_range = range(1, 100, 1)
    k_range = range(2, 11, 1)
    lr_range = np.arange(1e-5, 1e-4, 1e-5)

    res = NN.NN_binary_scanner(epoch_range, k_range, lr_range, patterns, labels, label0, label1)
    
    if not os.path.exists(f'NN Not PCA'):
        os.makedirs(f'NN Not PCA')
    
    if not os.path.exists(f'NN Not PCA/score_heatmaps'):
        os.makedirs(f'NN Not PCA/score_heatmaps')
    
    if not os.path.exists(f'NN Not PCA/loss_heatmaps'):
        os.makedirs(f'NN Not PCA/loss_heatmaps')
    
    sens = res['Sens List']
    spec = res['Spec List']
    loss = res['Loss List']
    scores = np.concatenate((sens, spec), axis = 0)
    normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))
    loss_norm = colors.Normalize(vmin=np.min(loss), vmax=np.max(loss))

    for i,k in enumerate(k_range):
        fig, ax = plt.subplots(1,2, figsize=(16,9))

        sens_colormesh = NN.heatmap_plotter(ax[0], lr_range, epoch_range, sens[:,i,:], "Sensitivity", 'Learning Rate', 'Number of epochs', normalization, cm.viridis)
        spec_colormesh = NN.heatmap_plotter(ax[1], lr_range, epoch_range, spec[:,i,:], "Specificity", 'Learning Rate', 'Number of epochs', normalization, cm.viridis)
        
        fig.suptitle(f'{k}-Fold')
        fig.colorbar(spec_colormesh,  orientation='vertical')
        fig.savefig(f'NN Not PCA/score_heatmaps/heatmap_{k}_fold.png', dpi=120)
        
        fig_loss, ax_loss = plt.subplots( figsize=(16,9))

        loss_colormesh = NN.heatmap_plotter(ax_loss, lr_range, epoch_range, loss[:,i,:], 'Loss', 'Learning Rate', 'Number of epochs', loss_norm, cm.viridis )

        fig_loss.suptitle(f'{k}-Fold')
        fig_loss.colorbar(loss_colormesh,  orientation='vertical')
        fig_loss.savefig(f'NN Not PCA/loss_heatmaps/heatmap_{k}_fold.png', dpi=120)

    print("CPU Time:" + str(round((clock.process_time() - start_cpu)/60)) + "'" + str(round((clock.process_time() - start_cpu)%60)) + "''")
    plt.show()

    print('Si vuole fare altri grafici?\nPermere y per continuare\nPremere n per uscire')
    graph = input()    
    
    while graph != 'n':
        print('Scegliere quale fold graficare')
        print([f'{k} folds-->{i}' for i,k in enumerate(k_range)] )
        i_k = int(input())

        
        print('Scegliere quale epoch fissare')
        print([f'{ep} epochs-->{i}' for i,ep in enumerate(epoch_range)] )
        i_ep = int(input())

        print('Scegliere quale learning rate fissare')
        print([f'{lr:0.5f} LR-->{i}' for i,lr in enumerate(lr_range)] )
        i_lr = int(input())

        fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=(16,9))
        fig1.subplots_adjust(0.2, 0.2)
        ax1[0].scatter(lr_range, sens[i_ep,i_k,:], color='C1')
        ax1[0].errorbar(lr_range, sens[i_ep,i_k,:], yerr=np.std(sens[i_ep,i_k,:]), color='C1')
        ax1[0].grid(True)
        ax1[0].set_ylim(0, 1.1)
        ax1[0].set_title('Sensitivity')

        ax1[1].scatter(lr_range, spec[i_ep,i_k,:], color='C2')
        ax1[1].errorbar(lr_range, spec[i_ep,i_k,:], yerr=np.std(spec[i_ep,i_k,:]), color='C2')
        ax1[1].grid(True)
        ax1[1].set_ylim(0, 1.1)
        ax1[1].set_title('Specificity')

        ax1[1].set_xlabel('Learning rate')
        fig1.suptitle(f'Metrics: {k_range[i_k]}-folds, {epoch_range[i_ep]} epochs')
        fig1.savefig(f'NN Not PCA/lr_vs_metrics_{k_range[i_k]}_folds_{epoch_range[i_ep]}_epochs.png', dpi=120)
                

        fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(16,9))
        fig2.subplots_adjust(0.2, 0.2)
        ax2[0].scatter(epoch_range, sens[:,i_k,i_lr], color='C1')
        ax2[0].errorbar(epoch_range, sens[:,i_k,i_lr], yerr=np.std(sens[:,i_k,i_lr]), color='C1')
        ax2[0].grid(True)
        ax2[0].set_ylim(0, 1.1)
        ax2[0].set_title('Sensitivity')

        ax2[1].scatter(epoch_range, spec[:,i_k,i_lr], color='C2')
        ax2[1].errorbar(epoch_range, spec[:,i_k,i_lr], yerr=np.std(sens[:,i_k,i_lr]), color='C2')
        ax2[1].grid(True)
        ax2[1].set_ylim(0, 1.1)
        ax2[1].set_title('Specificity')

        ax2[1].set_xlabel('Epochs')
        fig2.suptitle(f'Metrics: {k_range[i_k]}-folds, {lr_range[i_lr]:0.5f} LR')
        fig2.savefig(f'NN Not PCA/epoch_vs_metrics_{k_range[i_k]}_folds_{lr_range[i_lr]:0.5f}_LR.png', dpi=120)

        fig3, ax3 = plt.subplots(2, 1, figsize=(16,9))
        fig3.subplots_adjust(hspace=0.5)
        ax3[0].plot(epoch_range, loss[:,i_k,i_lr], color='C1', label='CV Loss')
        # ax3[0].errorbar(epoch_range, loss[:,i_k,i_lr], yerr=np.std(loss[:,i_k,i_lr]), color='C1')
        ax3[0].grid(True)
        ax3[0].set_ylim(np.min(loss[:,i_k,i_lr]), np.max(loss[:,i_k,i_lr]))
        ax3[0].set_title(f'Loss at {lr_range[i_lr]:0.5f} LR fixed')
        ax3[0].legend()
        ax3[0].set_xlabel('Epoch')

        ax3[1].plot(lr_range, loss[i_ep,i_k,:], color='C2', label='CV Loss')
        # ax3[1].errorbar(lr_range, loss[i_ep,i_k,:], yerr=np.std(loss[i_ep,i_k,:]), color='C2')
        ax3[1].grid(True)
        ax3[1].set_ylim(np.min(loss[i_ep,i_k,:]), np.max(loss[i_ep,i_k,:]))
        ax3[1].set_title(f'Loss at {epoch_range[i_ep]} epochs fixed')
        ax3[1].legend()
        ax3[1].set_xlabel('Learning Range')

        fig3.suptitle('Loss Function')
        fig3.savefig(f'NN Not PCA/loss.png', dpi=120)


        plt.show()

        print('Si vuole fare altri grafici?\nPermere y per continuare\nPremere n per uscire')
        graph = input()  