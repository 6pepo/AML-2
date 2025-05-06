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