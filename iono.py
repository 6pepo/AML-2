from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
  
# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
x = ionosphere.data.features 
y = ionosphere.data.targets 

x = x.to_numpy()
y = y.to_numpy()


# Mask per ciascuna label
mask_good = y == 'g'
# Trova gli indici dove y è 'g' o 'b'
idx_good = np.where(y == 'g')[0]
idx_bad = np.where(y == 'b')[0]

# Usa gli indici per estrarre da x
x_good = x[idx_good]
x_bad = x[idx_bad]

#print(len(x), len(x_good), len(x_bad))

print("x_good:", x_good)
print("x_bad:", x_bad)


print('Type x:', type(x))
print('Type y:', type(y))
print(x)
print(y)

fig, ax = plt.subplots()
ax.plot(x[0], 'b', label=y[0])
ax.plot(x[1], 'r', label=y[1])
plt.legend()

# metadata 
print('Metadata:\n', ionosphere.metadata) 
  
# variable information 
print('Variable info:\n', ionosphere.variables) 

plt.show()
