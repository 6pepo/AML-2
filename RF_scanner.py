import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import RF_Library as RF
import time as clock
import torch
import os

from ucimlrepo import fetch_ucirepo 
from matplotlib import cm, colors
  
start_cpu = clock.process_time()

# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
x = ionosphere.data.features 
y = ionosphere.data.targets 

# signal = np.zeros((x.shape[0], x.shape[1]//2), dtype=np.complex128)
# for i,rows in enumerate(x.values):
#     for k in range(0,len(rows)-1, 2):
#         signal[i,k//2] = rows[k] + 1j*rows[k+1]

signal = x.to_numpy()
labels = y.to_numpy().ravel()

tree_range = range(100,401,100)
k_range = range(2,11,3)
n_seed = 3

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

res, indexes = RF.RF_binary_scanner_random_pick(tree_range, k_range, n_seed, signal, labels, label0, label1, pick_numb=len(signal)//3)

sensitivity = res['Sens List']
sensitivity_std = res['Sens Std List']
specificity = res['Spec List']
specificity_std = res['Spec Std List']


if not os.path.exists(f'{n_seed} Random Seeds Not PCA'):
        os.makedirs(f'{n_seed} Random Seeds Not PCA')

fig, ax = plt.subplots(1,2, figsize=(16,9))
norm = colors.Normalize(vmin = 0, vmax = np.max(np.concatenate((sensitivity,specificity),axis=0)))
sens_colormesh = RF.heatmap_plotter(ax[0], x=k_range, y=tree_range, array=sensitivity, title="Sensitivity", norm=norm )
spec_colormesh = RF.heatmap_plotter(ax[1], x=k_range, y=tree_range, array=specificity, title="Specificity", norm=norm )
fig.colorbar(spec_colormesh,  orientation='vertical')
fig.savefig(f'{n_seed} Random Seeds Not PCA/heatmaps.png', dpi=120)

fig1, ax1 = plt.subplots(figsize=(16,9))
bin_vals, bins, _ = ax1.hist(indexes.ravel(), bins=len(signal), color='red', edgecolor='black')
fig1.savefig(f'{n_seed} Random Seeds Not PCA/picked_indexes_distributions.png', dpi=120)


for i,k in enumerate(k_range):
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(16,9))
    fig.subplots_adjust(0.2, 0.2)
    ax2[0].scatter(tree_range, sensitivity[:, i], color='C1')
    ax2[0].errorbar(tree_range, sensitivity[:, i], yerr=sensitivity_std[:, i], color='C1')
    ax2[0].grid(True)
    ax2[0].set_ylim(0, 1.1)
    ax2[0].set_title('Sensitivity')

    ax2[1].scatter(tree_range, specificity[:, i], color='C2')
    ax2[1].errorbar(tree_range, specificity[:, i], yerr=specificity_std[:, i], color='C2')
    ax2[1].grid(True)
    ax2[1].set_ylim(0, 1.1)
    ax2[1].set_title('Specificity')

    ax2[1].set_xlabel('Number of Trees')
    fig2.suptitle(f'{k} Folds')

    fig2.savefig(f'{n_seed} Random Seeds Not PCA/{k}_folds.png', dpi=120)

plt.show()