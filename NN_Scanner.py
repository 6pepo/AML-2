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
    
    epoch_range = range(7000, 100000, 1000)

    k_range = range(2, 11, 1)

    lr_range = np.arange(1e-5, 1e-4, 1e-5)

    res = NN.NN_binary_scanner(epoch_range, k_range, lr_range, patterns, labels, label0, label1)
    
    if not os.path.exists(f'NN Not PCA'):
        os.makedirs(f'NN Not PCA')
    
    # Single n_fold plot with slider
    for i,k in enumerate(k_range):
        fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(16,9))
        fig2.subplots_adjust(0.2, 0.2)
        ax2[0].scatter(epoch_range, res['Sens List'][:, i], color='C1')
        ax2[0].errorbar(epoch_range, res['Sens List'][:, i], yerr=res['Sens Std List'][:, 0], color='C1')
        ax2[0].grid(True)
        ax2[0].set_ylim(0, 1.1)
        ax2[0].set_title('Sensitivity')

        ax2[1].scatter(epoch_range, res['Spec List'][:, i], color='C2')
        ax2[1].errorbar(epoch_range, res['Spec List'][:, i], yerr=res['Spec Std List'][:, 0], color='C2')
        ax2[1].grid(True)
        ax2[1].set_ylim(0, 1.1)
        ax2[1].set_title('Specificity')
    
        ax2[1].set_xlabel('Number of Epochs')
        fig2.suptitle(f'{k} Folds')
    
        fig2.savefig(f'NN Not PCA/{k}_folds.png', dpi=120)

    # Heatmaps n_fold - n_trees
    fig, ax = plt.subplots(1,2, figsize=(16,9))

    scores = np.concatenate((res['Sens List'], res['Spec List']), axis = 0)
    normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))

    sens_colormesh = NN.heatmap_plotter(ax[0], k_range, epoch_range, res['Sens List'], "Sensitivity", normalization, cm.viridis)
    spec_colormesh = NN.heatmap_plotter(ax[1], k_range, epoch_range, res['Spec List'], "Specificity", normalization, cm.viridis)
    
    fig.colorbar(spec_colormesh,  orientation='vertical')
    fig.savefig(f'NN Not PCA/heatmaps.png', dpi=120)

    print("CPU Time:" + str(round((clock.process_time() - start_cpu)/60)) + "'" + str(round((clock.process_time() - start_cpu)%60)) + "''")
    plt.show()