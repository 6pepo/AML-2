from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import RF_Library as RF
import time as clock
import torch
  
start_cpu = clock.process_time()

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

fig, ax = plt.subplots(1,2)
ax[0].plot(signal[0].real, label=f'{labels[0]} Real Part')
ax[0].plot(signal[0].imag, label=f'{labels[0]} Imaginary Part')
ax[0].grid(True)
ax[0].legend()

ax[1].plot(signal[1].real, label=f'{labels[1]} Real Part')
ax[1].plot(signal[1].imag, label=f'{labels[1]} Imaginary Part')
ax[1].grid(True)
ax[1].legend()

patterns = x.to_numpy()
labels = y.to_numpy().ravel()
print(labels)

# Picking first 50 good and 50 bad patterns for training, the rest is for External Testing
ntrain = 100
patt_train = np.empty((ntrain, x.shape[1]))
lab_train = np.full(ntrain, '')
patt_ext = np.empty((x.shape[0]-ntrain, x.shape[1]))
lab_ext = np.full(x.shape[0]-ntrain, '')
good_t = 0
bad_t = 0
good_e = 0
bad_e = 0

for i, lab in enumerate(labels):
    if lab == 'g':
        if good_t < ntrain/2:
            patt_train[good_t + bad_t] = patterns[i]
            lab_train[good_t + bad_t] = lab
            good_t += 1
        else:
            patt_ext[good_e + bad_e] = patterns[i]
            lab_ext[good_e + bad_e] = lab
            good_e += 1
    if lab == 'b':
        if bad_t < ntrain/2:
            patt_train[good_t + bad_t] = patterns[i]
            lab_train[good_t + bad_t] = lab
            bad_t += 1
        else:
            patt_ext[good_e + bad_e] = patterns[i]
            lab_ext[good_e + bad_e] = lab
            bad_e += 1

patt_train = np.delete(patt_train, 1, axis=1)     # Removes column 1 since it's all 0
patt_ext = np.delete(patt_ext, 1, axis=1)       # Removes column 1 since it's all 0

corr = np.corrcoef(patt_train, rowvar=False)    # 2x Real Features
# corr = np.corrcoef(signal, rowvar=False)      # Complex Features
e_val, e_vec = RF.torch_eig(corr, var_type=torch.float64)
e_val = np.real(e_val)
e_vec = np.real(e_vec)

sort_index = np.argsort(np.abs(e_val))[::-1]       
e_val = e_val[sort_index]
e_vec = e_vec[:, sort_index]

e_val_sum = np.sum(e_val)
val_sum = 0
n_vec = 0
perc_tresh = 0.9            # Threshold of features at wich we cut
print('\nPercent\tCumulative')
for i,val in enumerate(e_val):
    val_sum += val
    print(round(val/e_val_sum * 100, 2), '%\t', np.round(val_sum/e_val_sum * 100, 2), '%')
    if val_sum/e_val_sum > perc_tresh:
        n_vec = i+1
        break

print('Numbers of principal components:', n_vec)

df_corr = pd.DataFrame(data = corr)
df_eval = pd.DataFrame(data = e_val)
df_evec = pd.DataFrame(data = e_vec[:, :n_vec])
df_corr.to_csv('corrMatrix.csv', sep = ',')
df_eval.to_csv('eigenvalues.csv', sep = ',')
df_evec.to_csv('eigenvectors.csv', sep = ',')

df_trainPatt = pd.DataFrame(data = patt_train)
df_trainLab = pd.DataFrame(data = lab_train)
df_extPatt = pd.DataFrame(data = patt_ext)
df_extLab = pd.DataFrame(data = lab_ext)
df_trainPatt.to_csv('iono_trainPatt.csv', sep = ',')
df_trainLab.to_csv('iono_trainLab.csv', sep = ',')
df_extPatt.to_csv('iono_extPatt.csv', sep = ',')
df_extLab.to_csv('iono_extLab.csv', sep = ',')

print("CPU Time:" + str(round((clock.process_time() - start_cpu)/60)) + "'" + str(round((clock.process_time() - start_cpu)%60)) + "''")

plt.show()
