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
    
    # tree_range = range(10, 410, 10)

    # k_range = range(2, 11, 1)

    n_seeds = 100

    #res = NN.NN_binary_scanner(size_range, tol_range, n_seeds, patterns, labels, label0, label1)

    k_range = range(2, 20, 1)

    accuracy_list = np.zeros(len(k_range))
    sensitivity_list = np.zeros(len(k_range))
    specificity_list = np.zeros(len(k_range))

    for i, k in enumerate(k_range):

        
        res = NN.RF_binary_kfold(k, patterns, labels, label0, label1)

        accuracy_list[i] = res['Acc']
        sensitivity_list[i] = res['Sens']
        specificity_list[i] = res['Spec']

    fig, ax = plt.subplots(1, 3)

    ax[0].scatter(k_range, accuracy_list)
    ax[0].set_title('Accuracy')

    ax[1].scatter(k_range, sensitivity_list)
    ax[1].set_title('Sensitivity')

    ax[2].scatter(k_range, specificity_list)
    ax[2].set_title('Specificity')

    plt.show()


    '''
    if not os.path.exists(f'{n_seeds} Random Seeds Not PCA'):
        os.makedirs(f'{n_seeds} Random Seeds Not PCA')
    
    # Single n_fold plot with slider
    for i,k in enumerate(k_range):
        fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(16,9))
        fig2.subplots_adjust(0.2, 0.2)
        ax2[0].scatter(tree_range, res['Sens List'][:, i], color='C1')
        ax2[0].errorbar(tree_range, res['Sens List'][:, i], yerr=res['Sens Std List'][:, 0], color='C1')
        ax2[0].grid(True)
        ax2[0].set_ylim(0, 1.1)
        ax2[0].set_title('Sensitivity')

        ax2[1].scatter(tree_range, res['Spec List'][:, i], color='C2')
        ax2[1].errorbar(tree_range, res['Spec List'][:, i], yerr=res['Spec Std List'][:, 0], color='C2')
        ax2[1].grid(True)
        ax2[1].set_ylim(0, 1.1)
        ax2[1].set_title('Specificity')
    
        ax2[1].set_xlabel('Number of Trees')
        fig2.suptitle(f'{k} Folds')
    
        fig2.savefig(f'{n_seeds} Random Seeds Not PCA/{k}_folds.png', dpi=120)

    # Heatmaps n_fold - n_trees
    fig, ax = plt.subplots(1,2, figsize=(16,9))

    scores = np.concatenate((res['Sens List'], res['Spec List']), axis = 0)
    normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))

    sens_colormesh = RF.heatmap_plotter(ax[0], k_range, tree_range, res['Sens List'], "Sensitivity", normalization, cm.viridis)
    spec_colormesh = RF.heatmap_plotter(ax[1], k_range, tree_range, res['Spec List'], "Specificity", normalization, cm.viridis)
    
    fig.colorbar(spec_colormesh,  orientation='vertical')
    fig.savefig(f'{n_seeds} Random Seeds Not PCA/heatmaps.png', dpi=120)

    print("CPU Time:" + str(round((clock.process_time() - start_cpu)/60)) + "'" + str(round((clock.process_time() - start_cpu)%60)) + "''")
    plt.show()
    '''