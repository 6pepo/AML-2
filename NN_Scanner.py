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
    
    epoch_range = range(10000, 100000, 2000)
    k_range = range(2, 11, 1)
    lr_range = np.arange(1e-4, 1e-3, 0.5e-4)

    res = NN.NN_binary_scanner(epoch_range, k_range, lr_range, patterns, labels, label0, label1)
    
    if not os.path.exists(f'NN Not PCA'):
        os.makedirs(f'NN Not PCA')
    
    if not os.path.exists(f'NN Not PCA/heatmaps'):
        os.makedirs(f'NN Not PCA/heatmaps')
    
    sens = res['Sens List']
    spec = res['Spec List']
    # Heatmaps n_fold - n_trees
    for i,k in enumerate(k_range):
        fig, ax = plt.subplots(1,2, figsize=(16,9))

        scores = np.concatenate((sens, spec), axis = 0)
        normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))

        sens_colormesh_ep = NN.heatmap_plotter(ax[0], lr_range, epoch_range, sens[:,i,:], "Sensitivity", 'Learning Rate', 'Number of epochs', normalization, cm.viridis)
        spec_colormesh_ep = NN.heatmap_plotter(ax[1], lr_range, epoch_range, spec[:,i,:], "Specificity", 'Learning Rate', 'Number of epochs', normalization, cm.viridis)
        
        fig.suptitle(f'{k}-Fold')
        fig.colorbar(spec_colormesh_ep,  orientation='vertical')
        fig.savefig(f'NN Not PCA/heatmaps/heatmap_{k}_fold.png', dpi=120)

    print("CPU Time:" + str(round((clock.process_time() - start_cpu)/60)) + "'" + str(round((clock.process_time() - start_cpu)%60)) + "''")
    plt.show()