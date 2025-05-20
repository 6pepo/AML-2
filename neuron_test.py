import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import RF_Library as ML
import time as clock
import os

epochs = 50000
learnRate = 0.01

# Training Set
train_patterns = pd.read_csv("iono_trainPatt.csv", header=0, index_col=0)
train_labels = pd.read_csv("iono_trainLab.csv", header=0, index_col=0)

train_patterns = train_patterns.to_numpy()
train_labels = train_labels.to_numpy().ravel()
label0 = 'b'
label1 = 'g'

# External test Set
ext_patterns = pd.read_csv("iono_extPatt.csv", header=0, index_col=0)
ext_labels = pd.read_csv("iono_extLab.csv", header=0, index_col=0)

ext_patterns = ext_patterns.to_numpy()
ext_labels = ext_labels.to_numpy().ravel()

good = 0
bad = 0

for l in train_labels:
    if l == label1:
        good += 1
    if l == label0:
        bad += 1

print('Good signals:', good)
print('Bad signals:', bad)

neuronSE = ML.neuron(labneg=label0, labpos=label1, max_epochs=epochs, learning_rate=learnRate, tol=1e-6, loss='squaredError')
neuronSE.fit(train_patterns, train_labels)
pred_SE = neuronSE.predict(ext_patterns)

print(f'Squared Error\nConvergence Epoch: {neuronSE.conv_epoch}')

confMatSE = np.zeros((2,2))
for i, pred in enumerate(pred_SE):
    if pred == label1:
        if ext_labels[i] == label1:
            confMatSE[0][0] += 1
        elif ext_labels[i] == label0:
            confMatSE[0][1] += 1
    elif pred == label0:
        if ext_labels[i] == label1:
            confMatSE[1][0] += 1
        elif ext_labels[i] == label0:
            confMatSE[1][1] += 1

print(confMatSE)
figSE, axSE = ML.confMat_binary_plot(confMatSE, title="Confusion matrix\nSquared Error")

fig2SE, as2SE = plt.subplots()
as2SE.plot(range(epochs), neuronSE.loss_list)
as2SE.set_xlabel('Epoch')
as2SE.set_ylabel('Loss')
as2SE.set_title('Squared Error')

neuronPERC = ML.neuron(labneg=label0, labpos=label1, max_epochs=epochs, learning_rate=learnRate, tol=1e-6, loss='perceptron')
neuronPERC.fit(train_patterns, train_labels)
pred_PERC = neuronPERC.predict(ext_patterns)

print(f'Perceptron\nConvergence Epoch: {neuronSE.conv_epoch}')

confMatPERC = np.zeros((2,2))
for i, pred in enumerate(pred_PERC):
    if pred == label1:
        if ext_labels[i] == label1:
            confMatPERC[0][0] += 1
        elif ext_labels[i] == label0:
            confMatPERC[0][1] += 1
    elif pred == label0:
        if ext_labels[i] == label1:
            confMatPERC[1][0] += 1
        elif ext_labels[i] == label0:
            confMatPERC[1][1] += 1

print(confMatPERC)
figPERC, axPERC = ML.confMat_binary_plot(confMatPERC, title="Confusion matrix\nPerceptron")

fig2PERC, as2PERC = plt.subplots()
as2PERC.plot(range(epochs), neuronPERC.loss_list)
as2PERC.set_xlabel('Epoch')
as2PERC.set_ylabel('Loss')
as2PERC.set_title('Perceptron')

plt.show()