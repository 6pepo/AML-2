import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import NN_Library as NN
import time as clock
import os

from matplotlib import cm, colors

if __name__ == '__main__':

    start_cpu = clock.process_time()

    pc_mat = pd.read_csv("eigenvectors.csv", sep=',', index_col=0)
    pc_mat = pc_mat.to_numpy()

    patterns = pd.read_csv("iono_trainPatt.csv", header=0, index_col=0)
    patterns = patterns.to_numpy()
    patterns = patterns.dot(pc_mat)
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
    
    epoch_range = range(10000, 100000, 5000)
    k_range = range(2, 11, 1)
    lr_range = np.arange(1e-4, 1e-3, 0.5e-4)

    res = NN.NN_binary_scanner(epoch_range, k_range, lr_range, patterns, labels, label0, label1)
    
    if not os.path.exists(f'NN PCA'):
        os.makedirs(f'NN PCA')
    
    if not os.path.exists(f'NN PCA/heatmaps'):
        os.makedirs(f'NN PCA/heatmaps')
    
    sens = res['Sens List']
    spec = res['Spec List']
    scores = np.concatenate((sens, spec), axis = 0)
    normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))

    for i,k in enumerate(k_range):
        fig, ax = plt.subplots(1,2, figsize=(16,9))

        sens_colormesh_ep = NN.heatmap_plotter(ax[0], lr_range, epoch_range, sens[:,i,:], "Sensitivity", 'Learning Rate', 'Number of epochs', normalization, cm.viridis)
        spec_colormesh_ep = NN.heatmap_plotter(ax[1], lr_range, epoch_range, spec[:,i,:], "Specificity", 'Learning Rate', 'Number of epochs', normalization, cm.viridis)
        
        fig.suptitle(f'{k}-Fold')
        fig.colorbar(spec_colormesh_ep,  orientation='vertical')
        fig.savefig(f'NN PCA/heatmaps/heatmap_{k}_fold.png', dpi=120)

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
        fig1.savefig(f'NN PCA/lr_vs_metrics_{k_range[i_k]}_folds_{epoch_range[i_ep]}_epochs.png', dpi=120)
                

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
        fig2.savefig(f'NN PCA/epoch_vs_metrics_{k_range[i_k]}_folds_{lr_range[i_lr]:0.5f}_LR.png', dpi=120)


        plt.show()

        print('Si vuole fare altri grafici?\nPermere y per continuare\nPremere n per uscire')
        graph = input()  