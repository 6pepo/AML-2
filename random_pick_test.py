import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import RF_Library as RF
import time as clock
import torch

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

tree_range = range(100,400,50)
k_range = range(2,11,1)
n_seed = 2

label0 = 'b'
label1 = 'g'

res = RF.RF_binary_scanner_random_pick(tree_range, k_range, n_seed, signal, labels, label0, label1, pick_numb=len(signal)//3)
# res = RF.RF_binary_scanner(tree_range, k_range, n_seed, signal, labels, label0, label1)

sensitivity = res['Sens List']
specificity = res['Spec List']

fig, ax = plt.subplots(1,2)

norm = colors.Normalize(vmin = 0, vmax = np.max(np.concatenate((sensitivity,specificity),axis=0)))
sens_colormesh = RF.heatmap_plotter(ax[0], x=k_range, y=tree_range, array=sensitivity, title="Sensitivity", norm=norm )
spec_colormesh = RF.heatmap_plotter(ax[1], x=k_range, y=tree_range, array=specificity, title="Specificity", norm=norm )

fig.colorbar(spec_colormesh,  orientation='vertical')

plt.show()