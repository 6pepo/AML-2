import sys
import os

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as clock
import RandomForest.RF_Library as RF
import torch

from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
  
start_cpu = clock.process_time()

script_directory = os.path.dirname(os.path.abspath(__file__))
mother_directory = os.path.dirname(script_directory)
dataset_path = os.path.join(mother_directory, 'Dataset')

# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
x = ionosphere.data.features 
y = ionosphere.data.targets 

# Data as complex numbers (DISCARDED)
signal = np.zeros((x.shape[0], x.shape[1]//2), dtype=np.complex128)

for i,rows in enumerate(x.values):
    for k in range(0,len(rows)-1, 2):
        signal[i,k//2] = rows[k] + 1j*rows[k+1]

labels = y.to_numpy().ravel()

good_indexes = np.argwhere(labels == 'g')
good_signals = signal[good_indexes]
bad_signals = np.delete(signal, good_indexes, axis=0)

fig, ax = plt.subplots(1,2)
ax[0].plot(good_signals[0].real, label=f'Real Part')
ax[0].plot(good_signals[0].imag, label=f'Imaginary Part')
ax[0].plot(np.abs(good_signals[0]), label=f'Absolute Value')
ax[0].grid(True)
ax[0].set_title(f"Good Signal {0}")
ax[0].legend()

ax[1].plot(bad_signals[0].real, label=f'Real Part')
ax[1].plot(bad_signals[0].imag, label=f'Imaginary Part')
ax[1].plot(np.abs(bad_signals[0]), label=f'Absolute Value')
ax[1].grid(True)
ax[1].set_title(f"Bad Signal {0}")
ax[1].legend()

goodSlide = fig.add_axes([0.04, 0.05, 0.05, 0.75])
goodSlider = Slider(ax = goodSlide, label = "Good Signal", valmin=0, valmax=len(good_signals)-1, valstep = 1, valinit=0, orientation='vertical')
badSlide = fig.add_axes([0.93, 0.05, 0.05, 0.75])
badSlider = Slider(ax = badSlide, label = "Bad Signal", valmin=0, valmax=len(bad_signals)-1, valstep = 1, valinit=0, orientation='vertical')

def update(val):
    igood = int(goodSlider.val)
    ibad = int(badSlider.val)

    ax[0].clear()
    ax[0].plot(signal[igood].real, label=f'Real Part')
    ax[0].plot(signal[igood].imag, label=f'Imaginary Part')
    ax[0].plot(np.abs(signal[igood]), label=f'Absolute Value')
    ax[0].grid(True)
    ax[0].set_title(f"Good Signal {igood}")
    ax[0].legend()

    ax[1].clear()
    ax[1].plot(signal[ibad].real, label=f'Real Part')
    ax[1].plot(signal[ibad].imag, label=f'Imaginary Part')
    ax[1].plot(np.abs(signal[ibad]), label=f'Absolute Value')
    ax[1].grid(True)
    ax[1].set_title(f"Bad Signal {ibad}")
    ax[1].legend()

    fig.canvas.draw_idle()

goodSlider.on_changed(update)
badSlider.on_changed(update)

# Picking first 50 good and 50 bad patterns for training, the rest is for External Testing
ntrain = 100
patt_train = np.empty((ntrain, signal.shape[1]))
lab_train = np.full(ntrain, '')
patt_ext = np.empty((signal.shape[0]-ntrain, signal.shape[1]))
lab_ext = np.full(signal.shape[0]-ntrain, '')
good_t = 0
bad_t = 0
good_e = 0
bad_e = 0

for i, lab in enumerate(labels):
    if lab == 'g':
        if good_t < ntrain/2:
            patt_train[good_t + bad_t] = np.real(signal[i])
            lab_train[good_t + bad_t] = lab
            good_t += 1
        else:
            patt_ext[good_e + bad_e] = np.real(signal[i])
            lab_ext[good_e + bad_e] = lab
            good_e += 1
    if lab == 'b':
        if bad_t < ntrain/2:
            patt_train[good_t + bad_t] = np.real(signal[i])
            lab_train[good_t + bad_t] = lab
            bad_t += 1
        else:
            patt_ext[good_e + bad_e] = np.real(signal[i])
            lab_ext[good_e + bad_e] = lab
            bad_e += 1

print(f'Good signal: {good_t+good_e}, Bad signal: {bad_t+bad_e}')

corr = np.corrcoef(patt_train, rowvar=False)    
# e_val, e_vec = RF.torch_eig(corr, var_type=torch.float32)
e_val, e_vec = eigh(corr)
e_val = np.real(e_val)  #we cast to real because they have null imaginary part
e_vec = np.real(e_vec)

sort_index = np.argsort(np.abs(e_val))[::-1]       
e_val = e_val[sort_index]
e_vec = e_vec[:, sort_index]

e_val_sum = np.sum(e_val)
val_sum = 0
n_vec = 0
perc_tresh = 0.9            # Threshold of features at wich we cut
principal_components = np.zeros(len(e_val))
cum_percentage = np.zeros(len(e_val))
print('\nPercent\tCumulative')
for i, val in enumerate(e_val):
    val_sum += val
    print(round(val/e_val_sum * 100, 2), '%\t', np.round(val_sum/e_val_sum * 100, 2), '%')
    principal_components[i] = i+1
    cum_percentage[i] = np.round(val_sum/e_val_sum * 100, 2)
    if val_sum/e_val_sum > perc_tresh and n_vec==0:
        n_vec = i+1
        print('Threshold reached!')

fig1, ax1 = plt.subplots()
ax1.plot(principal_components,cum_percentage,'bo--' )
ax1.hlines(cum_percentage[n_vec-1], 0,16, colors='red')
ax1.plot(principal_components[n_vec-1], cum_percentage[n_vec-1], 'ro')
ax1.set_xlabel('N° Principal Compones')
ax1.set_ylabel('Cumulative Percentage')
ax1.grid(True)

print('Numbers of principal components:', n_vec)

df_corr = pd.DataFrame(data = corr)
df_eval = pd.DataFrame(data = e_val)
df_evec = pd.DataFrame(data = e_vec[:, :n_vec])
df_corr.to_csv(script_directory+'/corrMatrix.csv', sep = ',')
df_eval.to_csv(script_directory+'/eigenvalues.csv', sep = ',')
df_evec.to_csv(script_directory+'/eigenvectors.csv', sep = ',')

df_trainPatt = pd.DataFrame(data = patt_train)
df_trainLab = pd.DataFrame(data = lab_train)
df_extPatt = pd.DataFrame(data = patt_ext)
df_extLab = pd.DataFrame(data = lab_ext)
df_trainPatt.to_csv(script_directory+'/iono_trainPatt.csv', sep = ',')
df_trainLab.to_csv(script_directory+'/iono_trainLab.csv', sep = ',')
df_extPatt.to_csv(script_directory+'/iono_extPatt.csv', sep = ',')
df_extLab.to_csv(script_directory+'/iono_extLab.csv', sep = ',')

print("CPU Time:" + str(round((clock.process_time() - start_cpu)/60)) + "'" + str(round((clock.process_time() - start_cpu)%60)) + "''")

plt.show()
