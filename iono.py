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

signal = np.zeros((x.shape[0], x.shape[1]//2), dtype=np.complex128)

for i,rows in enumerate(x.values):
    for k in range(0,len(rows)-1, 2):
        signal[i,k//2] = rows[k] + 1j*rows[k+1]

y = y.to_numpy()


fig, ax = plt.subplots(1,2)
ax[0].plot(signal[0].real, label=f'{y[0]} Real Part')
ax[0].plot(signal[0].imag, label=f'{y[0]} Imaginary Part')
ax[0].grid(True)
ax[0].legend()

ax[1].plot(signal[1].real, label=f'{y[1]} Real Part')
ax[1].plot(signal[1].imag, label=f'{y[1]} Imaginary Part')
ax[1].grid(True)
ax[1].legend()

corr = np.corrcoef(signal, rowvar=False)
e_val, e_vec = RF.torch_eig(corr, var_type=torch.complex128)

sort_index = np.argsort(np.abs(e_val))[::-1]       
e_val = e_val[sort_index]
e_vec = e_vec[:, sort_index]

df_corr = pd.DataFrame(data = corr)
df_eval = pd.DataFrame(data = e_val)
df_evec = pd.DataFrame(data = e_vec)
df_corr.to_csv('corrMatrix.csv', sep = ',')
df_eval.to_csv('eigenvalues.csv', sep = ',')
df_evec.to_csv('eigenvectors.csv', sep = ',')

print("CPU Time:" + str(round((clock.process_time() - start_cpu)/60)) + "'" + str(round((clock.process_time() - start_cpu)%60)) + "''")

plt.show()