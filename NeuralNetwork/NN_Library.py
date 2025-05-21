import numpy as np
import sklearn.ensemble as ens
import matplotlib.pyplot as plt
import matplotlib.table as tab
import torch
import multiprocessing as mp
import os
import time as clock
import sys
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from scipy.special import expit     # Logistic Sigmoid
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from io import StringIO

def NN_binary_kfold(lr, k, n_ep, patterns, labels, label0, label1, iter = 0, ext_patt = None, ext_lab = None):
    start = clock.process_time()  
    models = []

    # Metrics of single folds
    fold_accuracy = []
    fold_sensitivity = []
    fold_specificity = []
    fold_precision = []
    fold_loss = []

    kf = StratifiedKFold(n_splits = k, shuffle = True)
    indices = kf.split(patterns, labels)

    if np.any(ext_lab != None):
        vote_1_ext = np.zeros(len(ext_lab))
        vote_0_ext = np.zeros(len(ext_lab))

        N1_ext = 0
        N0_ext = 0
        for lab in ext_lab:
            if lab == label0:
                N0_ext +=1
            if lab == label1:
                N1_ext +=1

    for i, (train_index, test_index) in enumerate(indices):

        train_pattern = patterns[train_index]
        train_labels = labels[train_index]

        test_pattern = patterns[test_index]
        test_labels = labels[test_index]

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        model = SGDClassifier(loss='perceptron',
                              penalty='l2', 
                              alpha=0.0001, 
                              l1_ratio=0.15, 
                              fit_intercept=True, 
                              max_iter=n_ep, 
                              tol=1e-6, 
                              shuffle=True, 
                              verbose=1, 
                              epsilon=0.1, 
                              n_jobs=None, 
                              random_state=None, 
                              learning_rate='constant', 
                              eta0=lr, 
                              power_t=0.5, 
                              early_stopping=False, 
                              validation_fraction=0.1, 
                              n_iter_no_change=5, 
                              class_weight=None, 
                              warm_start=True, 
                              average=False)

        model.fit(train_pattern, train_labels)
        models.append(model)

        sys.stdout = old_stdout
        loss_history = mystdout.getvalue()

        for line in loss_history.split('\n'):
            if(len(line.split("loss: ")) == 1):
                continue
            fold_loss.append(float(line.split("loss: ")[-1]))

        # Internal test
        test_prediction = model.predict(test_pattern)
        temp_confMat = np.zeros((2,2))

        for j, pred in enumerate(test_prediction):
            if pred == label1:
                if test_labels[j] == label1:
                    temp_confMat[0][0] += 1
                elif test_labels[j] == label0:
                    temp_confMat[0][1] += 1
            elif pred == label0:
                if test_labels[j] == label1:
                    temp_confMat[1][0] += 1
                elif test_labels[j] == label0:
                    temp_confMat[1][1] += 1

        temp_accuracy = (temp_confMat[0][0] + temp_confMat[1][1])/(temp_confMat[0][0] + temp_confMat[1][0] + temp_confMat[1][1] + temp_confMat[0][1])
        temp_sensitivity = (temp_confMat[0][0]) / (temp_confMat[0][0] + temp_confMat[1][0])
        temp_specificity = (temp_confMat[1][1]) / (temp_confMat[1][1] + temp_confMat[0][1])
        temp_precision = (temp_confMat[0][0]) / (temp_confMat[0][0] + temp_confMat[0][1]) if ((temp_confMat[0][0] + temp_confMat[0][1])) != 0 else 0

        fold_accuracy.append(temp_accuracy)
        fold_sensitivity.append(temp_sensitivity)
        fold_specificity.append(temp_specificity)
        fold_precision.append(temp_precision)

        # External test: Majority vote counting
        if np.any(ext_lab != None):
            ext_pred = model.predict(ext_patt)

            for j, pred in enumerate(ext_pred):
                if pred == label0:
                    vote_0_ext[j] += 1
                if pred == label1:
                    vote_1_ext[j] += 1

    # Performance of External Test: Best in Training
    # Best Accuracy
    bacc_model = models[np.argmax(fold_accuracy)]
    bacc_confMat = np.zeros((2,2))
    bacc_accuracy = 0.
    bacc_sensitivity = 0.
    bacc_specificity = 0.
    bacc_precision = 0.

    if np.any(ext_lab != None):
        bacc_predict = bacc_model.predict(ext_patt)

        for j, pred in enumerate(bacc_predict):
            if pred == label1:
                if ext_lab[j] == label1:
                    bacc_confMat[0][0] += 1
                elif ext_lab[j] == label0:
                    bacc_confMat[0][1] += 1
            elif pred == label0:
                if ext_lab[j] == label1:
                    bacc_confMat[1][0] += 1
                elif ext_lab[j] == label0:
                    bacc_confMat[1][1] += 1
        
        bacc_accuracy = (bacc_confMat[0][0] + bacc_confMat[1][1])/(bacc_confMat[0][0] + bacc_confMat[1][0] + bacc_confMat[1][1] + bacc_confMat[0][1])
        bacc_sensitivity = (bacc_confMat[0][0]) / (bacc_confMat[0][0] + bacc_confMat[1][0])
        bacc_specificity = (bacc_confMat[1][1]) / (bacc_confMat[1][1] + bacc_confMat[0][1])
        bacc_precision = (bacc_confMat[0][0]) / (bacc_confMat[0][0] + bacc_confMat[0][1]) if (bacc_confMat[0][0] + bacc_confMat[0][1]) != 0 else 0

    # Best Sensitivity
    bsens_model = models[np.argmax(fold_sensitivity)]
    bsens_confMat = np.zeros((2,2))
    bsens_accuracy = 0.
    bsens_sensitivity = 0.
    bsens_specificity = 0.
    bsens_precision = 0.

    if np.any(ext_lab != None):
        bsens_predict = bsens_model.predict(ext_patt)

        for j, pred in enumerate(bsens_predict):
            if pred == label1:
                if ext_lab[j] == label1:
                    bsens_confMat[0][0] += 1
                elif ext_lab[j] == label0:
                    bsens_confMat[0][1] += 1
            elif pred == label0:
                if ext_lab[j] == label1:
                    bsens_confMat[1][0] += 1
                elif ext_lab[j] == label0:
                    bsens_confMat[1][1] += 1
        
        bsens_accuracy = (bsens_confMat[0][0] + bsens_confMat[1][1])/(bsens_confMat[0][0] + bsens_confMat[1][0] + bsens_confMat[1][1] + bsens_confMat[0][1])
        bsens_sensitivity = (bsens_confMat[0][0]) / (bsens_confMat[0][0] + bsens_confMat[1][0])
        bsens_specificity = (bsens_confMat[1][1]) / (bsens_confMat[1][1] + bsens_confMat[0][1])
        bsens_precision = (bsens_confMat[0][0]) / (bsens_confMat[0][0] + bsens_confMat[0][1]) if (bsens_confMat[0][0] + bsens_confMat[0][1]) != 0 else 0

    # Best Specificity
    bspec_model = models[np.argmax(fold_specificity)]
    bspec_confMat = np.zeros((2,2))
    bspec_accuracy = 0.
    bspec_sensitivity = 0.
    bspec_specificity = 0.
    bspec_precision = 0.

    if np.any(ext_lab != None):
        bspec_predict = bspec_model.predict(ext_patt)

        for j, pred in enumerate(bspec_predict):
            if pred == label1:
                if ext_lab[j] == label1:
                    bspec_confMat[0][0] += 1
                elif ext_lab[j] == label0:
                    bspec_confMat[0][1] += 1
            elif pred == label0:
                if ext_lab[j] == label1:
                    bspec_confMat[1][0] += 1
                elif ext_lab[j] == label0:
                    bspec_confMat[1][1] += 1
        
        bspec_accuracy = (bspec_confMat[0][0] + bspec_confMat[1][1])/(bspec_confMat[0][0] + bspec_confMat[1][0] + bspec_confMat[1][1] + bspec_confMat[0][1])
        bspec_sensitivity = (bspec_confMat[0][0]) / (bspec_confMat[0][0] + bspec_confMat[1][0])
        bspec_specificity = (bspec_confMat[1][1]) / (bspec_confMat[1][1] + bspec_confMat[0][1])
        bspec_precision = (bspec_confMat[0][0]) / (bspec_confMat[0][0] + bspec_confMat[0][1]) if (bspec_confMat[0][0] + bspec_confMat[0][1]) != 0 else 0

    # Best Precision
    bprec_model = models[np.argmax(fold_precision)]
    bprec_confMat = np.zeros((2,2))
    bprec_accuracy = 0.
    bprec_sensitivity = 0.
    bprec_specificity = 0.
    bprec_precision = 0.

    if np.any(ext_lab != None):
        bprec_predict = bprec_model.predict(ext_patt)

        for j, pred in enumerate(bprec_predict):
            if pred == label1:
                if ext_lab[j] == label1:
                    bprec_confMat[0][0] += 1
                elif ext_lab[j] == label0:
                    bprec_confMat[0][1] += 1
            elif pred == label0:
                if ext_lab[j] == label1:
                    bprec_confMat[1][0] += 1
                elif ext_lab[j] == label0:
                    bprec_confMat[1][1] += 1
        
        bprec_accuracy = (bprec_confMat[0][0] + bprec_confMat[1][1])/(bprec_confMat[0][0] + bprec_confMat[1][0] + bprec_confMat[1][1] + bprec_confMat[0][1])
        bprec_sensitivity = (bprec_confMat[0][0]) / (bprec_confMat[0][0] + bprec_confMat[1][0])
        bprec_specificity = (bprec_confMat[1][1]) / (bprec_confMat[1][1] + bprec_confMat[0][1])
        bprec_precision = (bprec_confMat[0][0]) / (bprec_confMat[0][0] + bprec_confMat[0][1]) if (bprec_confMat[0][0] + bprec_confMat[0][1]) != 0 else 0

    # External test: Majority vote
    ext_confMat = np.zeros((2,2))
    ext_accuracy = 0.
    ext_sensitivity = 0.
    ext_specificity = 0.
    ext_precision = 0.

    if np.any(ext_lab != None):
        for i, true_label in enumerate(ext_lab):
            if vote_0_ext[i]>=vote_1_ext[i]:
                if true_label == label1:
                    ext_confMat[1][0] += 1
                if true_label == label0:
                    ext_confMat[1][1] += 1
            if vote_0_ext[i]<vote_1_ext[i]:
                if true_label == label1:
                    ext_confMat[0][0] += 1
                if true_label == label0:
                    ext_confMat[0][1] += 1
    
        ext_accuracy = (ext_confMat[0][0] + ext_confMat[1][1])/(ext_confMat[0][0] + ext_confMat[1][0] + ext_confMat[1][1] + ext_confMat[0][1])
        ext_sensitivity = (ext_confMat[0][0]) / (ext_confMat[0][0] + ext_confMat[1][0])
        ext_specificity = (ext_confMat[1][1]) / (ext_confMat[1][1] + ext_confMat[0][1])
        ext_precision = (ext_confMat[0][0]) / (ext_confMat[0][0] + ext_confMat[0][1]) if (ext_confMat[0][0] + ext_confMat[0][1]) != 0 else 0

    cpu_time_stamp = clock.process_time() - start

    res = {
        'learning_rate': lr,
        'k': k,
        'epoch': n_ep,
        'Acc': np.mean(fold_accuracy),
        'Acc Err': np.std(fold_accuracy),    #/np.sqrt(k),
        'Sens': np.mean(fold_sensitivity),
        'Sens Err': np.std(fold_sensitivity),    #/np.sqrt(k),
        'Spec': np.mean(fold_specificity),
        'Spec Err': np.std(fold_specificity),    #/np.sqrt(k),
        'Prec': np.mean(fold_precision),
        'Prec Err': np.std(fold_precision),
        'Loss': np.mean(fold_loss),
        'Ext Acc': ext_accuracy,
        'Ext Sens': ext_sensitivity,
        'Ext Spec': ext_specificity,
        'Ext Prec': ext_precision,
        'BAcc Acc': bacc_accuracy,
        'BAcc Sens': bacc_sensitivity,
        'BAcc Spec': bacc_specificity,
        'BAcc Prec': bacc_precision,
        'BSens Acc': bsens_accuracy,
        'BSens Sens': bsens_sensitivity,
        'BSens Spec': bsens_specificity,
        'BSens Prec': bsens_precision,
        'BSpec Acc': bspec_accuracy,
        'BSpec Sens': bspec_sensitivity,
        'BSpec Spec': bspec_specificity,
        'BSpec, Prec': bspec_precision,
        'BPrec Acc': bprec_accuracy,
        'BPrec Sens': bprec_sensitivity,
        'BPrec Spec': bprec_specificity,
        'BPrec, Prec': bprec_precision,
        'Models': models,
        'Conf Mat': ext_confMat,
        'Time': cpu_time_stamp
    }

    return res

def Neuron_binary_kfold(lr, k, n_ep, patterns, labels, label0, label1, iter = 0, ext_patt = None, ext_lab = None):
    start = clock.process_time()  
    models = []

    # Metrics of single folds
    fold_accuracy = []
    fold_sensitivity = []
    fold_specificity = []
    fold_precision = []
    fold_loss = []

    kf = StratifiedKFold(n_splits = k, shuffle = True)
    indices = kf.split(patterns, labels)

    if np.any(ext_lab != None):
        vote_1_ext = np.zeros(len(ext_lab))
        vote_0_ext = np.zeros(len(ext_lab))

        N1_ext = 0
        N0_ext = 0
        for lab in ext_lab:
            if lab == label0:
                N0_ext +=1
            if lab == label1:
                N1_ext +=1

    for i, (train_index, test_index) in enumerate(indices):

        train_pattern = patterns[train_index]
        train_labels = labels[train_index]

        test_pattern = patterns[test_index]
        test_labels = labels[test_index]

        model = neuron(labneg=label0, labpos=label1, max_epochs=n_ep, learning_rate=lr, tol=1e-6)


        model.fit(train_pattern, train_labels)
        models.append(model)

        fold_loss.append(model.loss_list)

        # Internal test
        test_prediction = model.predict(test_pattern)
        temp_confMat = np.zeros((2,2))

        for j, pred in enumerate(test_prediction):
            if pred == label1:
                if test_labels[j] == label1:
                    temp_confMat[0][0] += 1
                elif test_labels[j] == label0:
                    temp_confMat[0][1] += 1
            elif pred == label0:
                if test_labels[j] == label1:
                    temp_confMat[1][0] += 1
                elif test_labels[j] == label0:
                    temp_confMat[1][1] += 1

        temp_accuracy = (temp_confMat[0][0] + temp_confMat[1][1])/(temp_confMat[0][0] + temp_confMat[1][0] + temp_confMat[1][1] + temp_confMat[0][1])
        temp_sensitivity = (temp_confMat[0][0]) / (temp_confMat[0][0] + temp_confMat[1][0])
        temp_specificity = (temp_confMat[1][1]) / (temp_confMat[1][1] + temp_confMat[0][1])
        temp_precision = (temp_confMat[0][0]) / (temp_confMat[0][0] + temp_confMat[0][1]) if ((temp_confMat[0][0] + temp_confMat[0][1])) != 0 else 0

        fold_accuracy.append(temp_accuracy)
        fold_sensitivity.append(temp_sensitivity)
        fold_specificity.append(temp_specificity)
        fold_precision.append(temp_precision)

        # External test: Majority vote counting
        if np.any(ext_lab != None):
            ext_pred = model.predict(ext_patt)

            for j, pred in enumerate(ext_pred):
                if pred == label0:
                    vote_0_ext[j] += 1
                if pred == label1:
                    vote_1_ext[j] += 1

    # Performance of External Test: Best in Training
    # Best Accuracy
    bacc_model = models[np.argmax(fold_accuracy)]
    bacc_confMat = np.zeros((2,2))
    bacc_accuracy = 0.
    bacc_sensitivity = 0.
    bacc_specificity = 0.
    bacc_precision = 0.

    if np.any(ext_lab != None):
        bacc_predict = bacc_model.predict(ext_patt)

        for j, pred in enumerate(bacc_predict):
            if pred == label1:
                if ext_lab[j] == label1:
                    bacc_confMat[0][0] += 1
                elif ext_lab[j] == label0:
                    bacc_confMat[0][1] += 1
            elif pred == label0:
                if ext_lab[j] == label1:
                    bacc_confMat[1][0] += 1
                elif ext_lab[j] == label0:
                    bacc_confMat[1][1] += 1
        
        bacc_accuracy = (bacc_confMat[0][0] + bacc_confMat[1][1])/(bacc_confMat[0][0] + bacc_confMat[1][0] + bacc_confMat[1][1] + bacc_confMat[0][1])
        bacc_sensitivity = (bacc_confMat[0][0]) / (bacc_confMat[0][0] + bacc_confMat[1][0])
        bacc_specificity = (bacc_confMat[1][1]) / (bacc_confMat[1][1] + bacc_confMat[0][1])
        bacc_precision = (bacc_confMat[0][0]) / (bacc_confMat[0][0] + bacc_confMat[0][1])

    # Best Sensitivity
    bsens_model = models[np.argmax(fold_sensitivity)]
    bsens_confMat = np.zeros((2,2))
    bsens_accuracy = 0.
    bsens_sensitivity = 0.
    bsens_specificity = 0.
    bsens_precision = 0.

    if np.any(ext_lab != None):
        bsens_predict = bsens_model.predict(ext_patt)

        for j, pred in enumerate(bsens_predict):
            if pred == label1:
                if ext_lab[j] == label1:
                    bsens_confMat[0][0] += 1
                elif ext_lab[j] == label0:
                    bsens_confMat[0][1] += 1
            elif pred == label0:
                if ext_lab[j] == label1:
                    bsens_confMat[1][0] += 1
                elif ext_lab[j] == label0:
                    bsens_confMat[1][1] += 1
        
        bsens_accuracy = (bsens_confMat[0][0] + bsens_confMat[1][1])/(bsens_confMat[0][0] + bsens_confMat[1][0] + bsens_confMat[1][1] + bsens_confMat[0][1])
        bsens_sensitivity = (bsens_confMat[0][0]) / (bsens_confMat[0][0] + bsens_confMat[1][0])
        bsens_specificity = (bsens_confMat[1][1]) / (bsens_confMat[1][1] + bsens_confMat[0][1])
        bsens_precision = (bsens_confMat[0][0]) / (bsens_confMat[0][0] + bsens_confMat[0][1])

    # Best Specificity
    bspec_model = models[np.argmax(fold_specificity)]
    bspec_confMat = np.zeros((2,2))
    bspec_accuracy = 0.
    bspec_sensitivity = 0.
    bspec_specificity = 0.
    bspec_precision = 0.

    if np.any(ext_lab != None):
        bspec_predict = bspec_model.predict(ext_patt)

        for j, pred in enumerate(bspec_predict):
            if pred == label1:
                if ext_lab[j] == label1:
                    bspec_confMat[0][0] += 1
                elif ext_lab[j] == label0:
                    bspec_confMat[0][1] += 1
            elif pred == label0:
                if ext_lab[j] == label1:
                    bspec_confMat[1][0] += 1
                elif ext_lab[j] == label0:
                    bspec_confMat[1][1] += 1
        
        bspec_accuracy = (bspec_confMat[0][0] + bspec_confMat[1][1])/(bspec_confMat[0][0] + bspec_confMat[1][0] + bspec_confMat[1][1] + bspec_confMat[0][1])
        bspec_sensitivity = (bspec_confMat[0][0]) / (bspec_confMat[0][0] + bspec_confMat[1][0])
        bspec_specificity = (bspec_confMat[1][1]) / (bspec_confMat[1][1] + bspec_confMat[0][1])
        bspec_precision = (bspec_confMat[0][0]) / (bspec_confMat[0][0] + bspec_confMat[0][1])

    # Best Precision
    bprec_model = models[np.argmax(fold_precision)]
    bprec_confMat = np.zeros((2,2))
    bprec_accuracy = 0.
    bprec_sensitivity = 0.
    bprec_specificity = 0.
    bprec_precision = 0.

    if np.any(ext_lab != None):
        bprec_predict = bprec_model.predict(ext_patt)

        for j, pred in enumerate(bprec_predict):
            if pred == label1:
                if ext_lab[j] == label1:
                    bprec_confMat[0][0] += 1
                elif ext_lab[j] == label0:
                    bprec_confMat[0][1] += 1
            elif pred == label0:
                if ext_lab[j] == label1:
                    bprec_confMat[1][0] += 1
                elif ext_lab[j] == label0:
                    bprec_confMat[1][1] += 1
        
        bprec_accuracy = (bprec_confMat[0][0] + bprec_confMat[1][1])/(bprec_confMat[0][0] + bprec_confMat[1][0] + bprec_confMat[1][1] + bprec_confMat[0][1])
        bprec_sensitivity = (bprec_confMat[0][0]) / (bprec_confMat[0][0] + bprec_confMat[1][0])
        bprec_specificity = (bprec_confMat[1][1]) / (bprec_confMat[1][1] + bprec_confMat[0][1])
        bprec_precision = (bprec_confMat[0][0]) / (bprec_confMat[0][0] + bprec_confMat[0][1])

    # External test: Majority vote
    ext_confMat = np.zeros((2,2))
    ext_accuracy = 0.
    ext_sensitivity = 0.
    ext_specificity = 0.
    ext_precision = 0.

    if np.any(ext_lab != None):
        for i, true_label in enumerate(ext_lab):
            if vote_0_ext[i]>=vote_1_ext[i]:
                if true_label == label1:
                    ext_confMat[1][0] += 1
                if true_label == label0:
                    ext_confMat[1][1] += 1
            if vote_0_ext[i]<vote_1_ext[i]:
                if true_label == label1:
                    ext_confMat[0][0] += 1
                if true_label == label0:
                    ext_confMat[0][1] += 1
    
        ext_accuracy = (ext_confMat[0][0] + ext_confMat[1][1])/(ext_confMat[0][0] + ext_confMat[1][0] + ext_confMat[1][1] + ext_confMat[0][1])
        ext_sensitivity = (ext_confMat[0][0]) / (ext_confMat[0][0] + ext_confMat[1][0])
        ext_specificity = (ext_confMat[1][1]) / (ext_confMat[1][1] + ext_confMat[0][1])
        ext_precision = (ext_confMat[0][0]) / (ext_confMat[0][0] + ext_confMat[0][1])

    cpu_time_stamp = clock.process_time() - start

    res = {
        'learning_rate': lr,
        'k': k,
        'epoch': n_ep,
        'Acc': np.mean(fold_accuracy),
        'Acc Err': np.std(fold_accuracy),    #/np.sqrt(k),
        'Sens': np.mean(fold_sensitivity),
        'Sens Err': np.std(fold_sensitivity),    #/np.sqrt(k),
        'Spec': np.mean(fold_specificity),
        'Spec Err': np.std(fold_specificity),    #/np.sqrt(k),
        'Prec': np.mean(fold_precision),
        'Prec Err': np.std(fold_precision),
        'Loss': np.mean(fold_loss),
        'Ext Acc': ext_accuracy,
        'Ext Sens': ext_sensitivity,
        'Ext Spec': ext_specificity,
        'Ext Prec': ext_precision,
        'BAcc Acc': bacc_accuracy,
        'BAcc Sens': bacc_sensitivity,
        'BAcc Spec': bacc_specificity,
        'BAcc Prec': bacc_precision,
        'BSens Acc': bsens_accuracy,
        'BSens Sens': bsens_sensitivity,
        'BSens Spec': bsens_specificity,
        'BSens Prec': bsens_precision,
        'BSpec Acc': bspec_accuracy,
        'BSpec Sens': bspec_sensitivity,
        'BSpec Spec': bspec_specificity,
        'BSpec, Prec': bspec_precision,
        'BPrec Acc': bprec_accuracy,
        'BPrec Sens': bprec_sensitivity,
        'BPrec Spec': bprec_specificity,
        'BPrec, Prec': bprec_precision,
        'Models': models,
        'Conf Mat': ext_confMat,
        'Time': cpu_time_stamp
    }

    return res

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)

def Neuron_binary_scanner(n_iter, max_epoch, k_range, lr_range, patterns, labels, label0, label1, ext_patt = None, ext_lab = None):
    len_k = len(k_range)
    len_lr = len(lr_range)

    accuracy_list = np.empty((len_k, len_lr, n_iter))
    sensitivity_list = np.empty((len_k, len_lr, n_iter))
    specificity_list = np.empty((len_k, len_lr, n_iter))
    precision_list = np.empty((len_k, len_lr, n_iter))
    loss_list = np.empty((len_k, len_lr, n_iter, max_epoch))

    print("Begin Scanning...")

    tot_iter = len_k * len_lr * n_iter
    progress = tqdm(total=tot_iter)
    iter = 0

    for i_lr, val_lr in enumerate(lr_range):
        for i_k, k in enumerate(k_range):
            for iter in range(n_iter):
                # Progress bar
                progress.set_description(f"Fold: {i_k+1}/{len_k} | Learning Rate: {val_lr:0.3e} | Iteration: {iter+1}\{n_iter}")
                progress.update(1)

                res = Neuron_binary_kfold(val_lr, k, max_epoch, patterns, labels, label0, label1)

                accuracy_list[i_k, i_lr, iter] = res['Acc']
                sensitivity_list[i_k, i_lr, iter] = res['Sens']
                specificity_list[i_k, i_lr, iter] = res['Spec']
                precision_list[i_k, i_lr, iter] = res['Prec']
                loss_list[i_k, i_lr, iter, :] = res['Loss']

    print("\nFinished Scanning!                                                \n")

    res = {
        'Acc List': np.mean(accuracy_list, axis=2),
        'Acc Std List': np.std(accuracy_list, axis=2),
        'Sens List': np.mean(sensitivity_list, axis=2),
        'Sens Std List': np.std(sensitivity_list, axis=2),
        'Spec List': np.mean(specificity_list, axis=2),
        'Spec Std List': np.std(specificity_list, axis=2),
        'Prec List': np.mean(precision_list, axis=2),
        'Prec Std List': np.std(precision_list, axis=2),
        'Loss List': np.mean(loss_list, axis=2),
        'Loss Std List': np.std(loss_list, axis=2)
    }

    return res

def NN_binary_scanner_iter(iter, epoch_range, k_range, lr_range, patterns, labels, label0, label1, ext_patt = None, ext_lab = None, from_scratch=False):
    len_ep = len(epoch_range)
    len_k = len(k_range)
    len_lr = len(lr_range)

    accuracy_list = np.empty((len_ep, len_k, len_lr, iter))
    sensitivity_list = np.empty((len_ep, len_k, len_lr, iter))
    specificity_list = np.empty((len_ep, len_k, len_lr, iter))
    precision_list = np.empty((len_ep, len_k, len_lr, iter))
    loss_list = np.empty((len_ep, len_k, len_lr, iter))

    print("Begin Scanning...")

    tot_iter = len_ep*len_k*len_lr*iter
    progress = tqdm(total=tot_iter)

    for i_lr, val_lr in enumerate(lr_range):
        for i_ep, n_ep in enumerate(epoch_range):
            for i_k, k in enumerate(k_range):
                for n in range(iter):
                    
                    # Progress bar
                    progress.set_description(f"Iteration: {n}/{iter} | Epoch: {n_ep} | Fold: {i_k+1}/{len_k} | Learning Rate: {val_lr:0.3e}")
                    progress.update(1)
                    
                    # print("N_trees: ", n_trees, "\tN_fold: ", k, "\tIteration: ", iter, "/", tot_iter,end='\r')     #Python 3.x
                    # print("N_trees: {}\tN_fold: {}\tIteration: {}/{} \r".format(n_trees, k, iter, tot_iter)),       #Python 2.x

                    if from_scratch == True:
                        res = Neuron_binary_kfold(val_lr, k, n_ep, patterns, labels, label0, label1)
                    if from_scratch == False:
                        res = NN_binary_kfold(val_lr, k, n_ep, patterns, labels, label0, label1)

                    accuracy_list[i_ep, i_k, i_lr, n] = res['Acc']
                    sensitivity_list[i_ep, i_k, i_lr, n] = res['Sens']
                    specificity_list[i_ep, i_k, i_lr, n] = res['Spec']
                    precision_list[i_ep, i_k, i_lr, n] = res['Prec']
                    loss_list[i_ep, i_k, i_lr, n] = res['Loss']

    print("\nFinished Scanning!                                                \n")

    res = {
        'Acc List': np.mean(accuracy_list, axis=3),
        'Sens List': np.mean(sensitivity_list, axis=3),
        'Spec List': np.mean(specificity_list, axis=3),
        'Prec List': np.mean(precision_list, axis=3),
        'Loss List': np.mean(loss_list, axis=3),
        'Acc Std List': np.std(accuracy_list, axis=3),
        'Sens Std List': np.std(sensitivity_list, axis=3),
        'Spec Std List': np.std(specificity_list, axis=3),
        'Prec Std List': np.std(precision_list, axis=3),
        'Loss Std List': np.std(loss_list, axis=3),
    }

    return res

def NN_kfold_par(k, par, patterns, labels, label0, label1, iter = 0, ext_patt = None, ext_lab = None):
    start = clock.process_time() 

    # NOTA: par deve deve essere del tipo:
    # par = {
    #         'loss': 'log_loss',
    #         'alpha': 0.0001,
    #         'max_iter': 10000,
    #         'tol': 0.00001,
    #         'learning_rate': 'constant',
    #         'eta0': 0.0,
    #     }

    # Parameter of SGDClassifier
    loss = par['loss']  # def: 'log_loss'
    alpha = par['alpha']    # def: 0.0001
    max_iter = par['max_iter']  # numero di epoche
    tol = par['tol']    # def: 0.00001 
    learning_rate = par['learning_rate']    # def: 'constant'
    eta0 = par['eta0']  # lr iniziale

    models = []

    # Metrics of single folds
    fold_accuracy = []
    fold_sensitivity = []
    fold_specificity = []

    kf = StratifiedKFold(n_splits = k, shuffle = True)
    indices = kf.split(patterns, labels)

    if np.any(ext_lab != None):
        vote_1_ext = np.zeros(len(ext_lab))
        vote_0_ext = np.zeros(len(ext_lab))

        N1_ext = 0
        N0_ext = 0
        for lab in ext_lab:
            if lab == label0:
                N0_ext +=1
            if lab == label1:
                N1_ext +=1

    for i, (train_index, test_index) in enumerate(indices):

        train_pattern = patterns[train_index]
        train_labels = labels[train_index]

        test_pattern = patterns[test_index]
        test_labels = labels[test_index]

        model = SGDClassifier(loss=loss, # def: 'log_loss'
                              penalty='l2', 
                              alpha=alpha, # def: 0.0001
                              l1_ratio=0.15, 
                              fit_intercept=True, 
                              max_iter=max_iter, # numero di epoche 
                              tol=tol, # def: 0.00001 
                              shuffle=True, 
                              verbose=0, 
                              epsilon=0.1, 
                              n_jobs=None, 
                              random_state=None, 
                              learning_rate=learning_rate, # def: 'constant'
                              eta0=eta0, # lr iniziale
                              power_t=0.5, 
                              early_stopping=False, 
                              validation_fraction=0.1, 
                              n_iter_no_change=5, 
                              class_weight=None, 
                              warm_start=True, 
                              average=False)


        model.fit(train_pattern, train_labels)
        models.append(model)

        # Internal test
        test_prediction = model.predict(test_pattern)

        temp_Npos = 0
        temp_Nneg = 0

        for t_lab in test_labels:
            if t_lab == label1:
                temp_Npos += 1
            if t_lab == label0:
                temp_Nneg += 1

        temp_accuracy = 0.
        temp_sensitivity = 0.
        temp_specificity = 0.

        for j, pred in enumerate(test_prediction):
            if pred == label1 and test_labels[j] == label1:
                temp_accuracy += 1./(temp_Npos+temp_Nneg)
                temp_sensitivity += 1./temp_Npos
    
            if pred == label0 and test_labels[j] == label0:
                temp_accuracy += 1./(temp_Npos+temp_Nneg)
                temp_specificity += 1./temp_Nneg

        fold_accuracy.append(temp_accuracy)
        fold_sensitivity.append(temp_sensitivity)
        fold_specificity.append(temp_specificity)

        # External test: Majority vote counting
        if np.any(ext_lab != None):
            ext_pred = model.predict(ext_patt)

            for j, pred in enumerate(ext_pred):
                if pred == label0:
                    vote_0_ext[j] += 1
                if pred == label1:
                    vote_1_ext[j] += 1

    # Performance of External Test: Best in Training
    # Best Accuracy
    bacc_model = models[np.argmax(fold_accuracy)]
    bacc_accuracy = 0.
    bacc_sensitivity = 0.
    bacc_specificity = 0.

    if np.any(ext_lab != None):
        bacc_predict = bacc_model.predict(ext_patt)

        for j, pred in enumerate(bacc_predict):
            if pred == label1 and ext_lab[j] == label1:
                bacc_accuracy += 1./(N1_ext+N0_ext)
                bacc_sensitivity += 1./N1_ext
        
            if pred == label0 and ext_lab[j] == label0:
                bacc_accuracy += 1./(N1_ext+N0_ext)
                bacc_specificity += 1./N0_ext

    # Best Sensitivity
    bsens_model = models[np.argmax(fold_sensitivity)]
    bsens_accuracy = 0.
    bsens_sensitivity = 0.
    bsens_specificity = 0.

    if np.any(ext_lab != None):
        bsens_predict = bsens_model.predict(ext_patt)

        for j, pred in enumerate(bsens_predict):
            if pred == label1 and ext_lab[j] == label1:
                bsens_accuracy += 1./(N1_ext+N0_ext)
                bsens_sensitivity += 1./N1_ext
        
            if pred == label0 and ext_lab[j] == label0:
                bsens_accuracy += 1./(N1_ext+N0_ext)
                bsens_specificity += 1./N0_ext

    # Best Specificity
    bspec_model = models[np.argmax(fold_specificity)]
    bspec_accuracy = 0.
    bspec_sensitivity = 0.
    bspec_specificity = 0.
    
    if np.any(ext_lab != None):
        bspec_predict = bspec_model.predict(ext_patt)

        for j, pred in enumerate(bspec_predict):
            if pred == label1 and ext_lab[j] == label1:
                bspec_accuracy += 1./(N1_ext+N0_ext)
                bspec_sensitivity += 1./N1_ext

        
            if pred == label0 and ext_lab[j] == label0:
                bspec_accuracy += 1./(N1_ext+N0_ext)
                bspec_specificity += 1./N0_ext

    # External test: Majority vote
    ext_accuracy = 0.
    ext_sensitivity = 0.
    ext_specificity = 0.
    ext_confMat = np.zeros((2,2))

    if np.any(ext_lab != None):
        for i, true_label in enumerate(ext_lab):
            if vote_0_ext[i]>=vote_1_ext[i]:
                if true_label == label1:
                    ext_confMat[1][0] += 1
                if true_label == label0:
                    ext_accuracy += 1./(N1_ext+N0_ext)
                    ext_specificity += 1./N0_ext
                    ext_confMat[1][1] += 1
            if vote_0_ext[i]<vote_1_ext[i]:
                if true_label == label1:
                    ext_accuracy += 1./(N1_ext+N0_ext)
                    ext_sensitivity += 1./N1_ext
                    ext_confMat[0][0] += 1
                if true_label == label0:
                    ext_confMat[0][1] += 1

    cpu_time_stamp = clock.process_time() - start

    loss = par['loss']  # def: 'log_loss'
    alpha = par['alpha']    # def: 0001
    max_iter = par['max_iter']  # numero di epoche
    tol = par['tol']    # def: 0.00001 
    learning_rate = par['learning_rate']    # def: 'constant'
    eta0 = par['eta0']  # lr iniziale

    res = {
        'k': k,

        'loss': loss,
        'alpha': alpha,
        'max_iter': max_iter,
        'tol': tol,
        'learning_rate': learning_rate,
        'eta0': eta0,

        'Acc': np.mean(fold_accuracy),
        'Acc Err': np.std(fold_accuracy),    #/np.sqrt(k),
        'Sens': np.mean(fold_sensitivity),
        'Sens Err': np.std(fold_sensitivity),    #/np.sqrt(k),
        'Spec': np.mean(fold_specificity),
        'Spec Err': np.std(fold_specificity),    #/np.sqrt(k),
        'Ext Acc': ext_accuracy,
        'Ext Sens': ext_sensitivity,
        'Ext Spec': ext_specificity,
        'BAcc Acc': bacc_accuracy,
        'BAcc Sens': bacc_sensitivity,
        'BAcc Spec': bacc_specificity,
        'BSens Acc': bsens_accuracy,
        'BSens Sens': bsens_sensitivity,
        'BSens Spec': bsens_specificity,
        'BSpec Acc': bspec_accuracy,
        'BSpec Sens': bspec_sensitivity,
        'BSpec Spec': bspec_specificity,
        'Models': models,
        'Conf Mat': ext_confMat,
        'Time': cpu_time_stamp
    }

    return res

def NN_parameters_scanner(k, par_all, par_range, patterns, labels, label0, label1, ext_patt = None, ext_lab = None):
    
    # NOTA: par_all deve deve essere del tipo:
    # par_all = {
    #         'loss': 'log_loss',
    #         'alpha': 0.0001,
    #         'max_iter': 10000,
    #         'tol': 0.00001,
    #         'learning_rate': 'constant',
    #         'eta0': None, # def: 0.0
    #     }
    # Uno dei valori di par_all deve essere None, sarà quello su cui sarà fatto lo scanner

    par_index = ['loss', 'alpha', 'max_iter', 'tol', 'learning_rate', 'eta0']
 
    sp = None   # inizializing scanning parameter
    for index in par_index:
        if par_all[index] is None:
            sp = index  # find the scanning parameter, varing in par_range
            break
    if sp is None:
        print('ERROR IN SCANNING PARAMETER')

    len_par = len(par_range)

    accuracy_list = np.empty((len_par, 100))
    sensitivity_list = np.empty((len_par, 100))
    specificity_list = np.empty((len_par, 100))

    
    print("Begin Scanning...")

    tot_iter = len_par*100
    progress = tqdm(total=tot_iter)
    iter = 0

    

    for i, val in enumerate(par_range):
        for j in range(1, 100, 1): # ciclo su 100 random state    
            # Progress bar
            progress.set_description(f"Iteration: {i+1}/{len_par} | {sp}: {val}")
            progress.update(1)

            par = par_all
            par_all[sp] = val

            res = NN_kfold_par(k, par, patterns, labels, label0, label1)

            accuracy_list[i, j] = res['Acc']
            sensitivity_list[i, j] = res['Sens']
            specificity_list[i, j] = res['Spec']

    print("\nFinished Scanning!                                                \n")

    res = {
        'Acc List': np.mean(accuracy_list, axis=1),
        'Acc Std List': np.std(accuracy_list, axis=1),
        'Sens List': np.mean(sensitivity_list, axis=1),
        'Sens Std List': np.std(sensitivity_list, axis=1),
        'Spec List': np.mean(specificity_list, axis=1),
        'Spec Std List': np.std(specificity_list, axis=1)
    }

    return res

# def NN_binary_scanner_MP(tree_range, k_range, n_seeds, patterns, labels, label0, label1, ext_patt = None, ext_lab = None):
    if __name__ == 'RF_Library':
        mp.freeze_support()

        len_tree = len(tree_range)
        len_k = len(k_range)

        accuracy_list = np.empty((len_tree, len_k, n_seeds))
        sensitivity_list = np.empty((len_tree, len_k, n_seeds))
        specificity_list = np.empty((len_tree, len_k, n_seeds))

        
        print("Begin Scanning...")

        tot_iter = len_tree*len_k*n_seeds

        NUMBER_OF_PROCESSES = int(os.cpu_count()/2)
        
        task_queue = mp.Queue()
        done_queue = mp.Queue()

        print("Preparing Tasks...")
        for a in tree_range:
            for b in k_range:
                for c in range(n_seeds):
                    task_queue.put((NN_binary_kfold, (a, b, patterns, labels, label0, label1, c)))

        for i in range(NUMBER_OF_PROCESSES):
            mp.Process(target=worker, args=(task_queue, done_queue)).start()

        progress = tqdm(total = tot_iter)
        progress.set_description("Executing Tasks")

        for i in range(tot_iter):
            res = done_queue.get()
            accuracy_list[tree_range.index(res['n trees']), k_range.index(res['k']), res['n seed']] = res['Acc']
            sensitivity_list[tree_range.index(res['n trees']), k_range.index(res['k']), res['n seed']] = res['Sens']
            specificity_list[tree_range.index(res['n trees']), k_range.index(res['k']), res['n seed']] = res['Spec']
            progress.update(1)

        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put('STOP')

        print("\nFinished Scanning!\n")

        res = {
            'Acc List': np.mean(accuracy_list, axis=2),
            'Acc Std List': np.std(accuracy_list, axis=2),
            'Sens List': np.mean(sensitivity_list, axis=2),
            'Sens Std List': np.std(sensitivity_list, axis=2),
            'Spec List': np.mean(specificity_list, axis=2),
            'Spec Std List': np.std(specificity_list, axis=2)
        }

        return res
    return 0

def confMat_binary_plot(conf_mat, accuracy=None, sensitivity=None, specificity=None, precision=None, title=None):
    fig, ax = plt.subplots()
    if title == None:
        ax.set_title("Confusion matrix")
    else:
        ax.set_title(title)
    ax.axis('off')
    fig.frameon = False
    
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat[0])):
            conf_mat[i,j] = round(conf_mat[i,j], 2)

    tot = conf_mat[0][0]+conf_mat[0][1]+conf_mat[1][0]+conf_mat[1][1]
    orig_pos = conf_mat[0][0] + conf_mat[1][0]
    orig_neg = conf_mat[0][1] + conf_mat[1][1]
    if precision == None:
        precision = conf_mat[0][0] / (conf_mat[0][0]+conf_mat[0][1])
    if sensitivity == None:
        sensitivity = conf_mat[0][0] / (orig_pos)
    if specificity == None:
        specificity = conf_mat[1][1] / (orig_neg)
    if accuracy == None:
        accuracy = (conf_mat[0][0] + conf_mat[1][1]) / tot
    F1score = 2. * (precision * sensitivity) / (precision + sensitivity)
    
    norm = colors.Normalize(vmin = 0, vmax = np.max(conf_mat))
    normalized = norm((0., conf_mat[0][0], conf_mat[0][1], conf_mat[1][0], conf_mat[1][1]))
    cell_color = cm.viridis(normalized)
    middle = np.max(conf_mat)/2
    col_max = np.argmax(conf_mat) + 1
    text_color = np.empty((2,2,4))
    for i in (0,1):
        for j in (0,1):
            if conf_mat[i][j] < middle:
                text_color[i][j] = cell_color[col_max]
            else:
                text_color[i][j] = cell_color[0]
                
    table = tab.Table(ax, loc='upper center')
    table.auto_set_font_size(False)
    fontproperties = {'family': 'sans-serif',
                      'style': 'normal',
                      'variant': 'normal',
                      'stretch': 'normal',
                      'weight': 'normal',
                      'size': 'medium',
                      'math_fontfamily': 'dejavusans'}

    table.add_cell(0,2, width = 0.3, height = 0.1, text='Actual', loc='right', fontproperties = fontproperties, facecolor = 'white')
    table[0,2].visible_edges = 'BTL'
    table.add_cell(0,3, width = 0.3, height = 0.1, text='Condition', loc='left', fontproperties = fontproperties, facecolor = 'white')
    table[0,3].visible_edges = 'BTR'

    table.add_cell(1,1, width = 0.2, height = 0.2, text=f'Total\n\n{tot}', loc='center', fontproperties = fontproperties)
    table.add_cell(1,2, width = 0.3, height = 0.2, text=f'Positive\n\n{(orig_pos)/tot:.2%}', loc='center', fontproperties = fontproperties, facecolor = 'white')
    table.add_cell(1,3, width = 0.3, height = 0.2, text=f'Negative\n\n{(orig_neg)/tot:.2%}', loc='center', fontproperties = fontproperties, facecolor = 'white')

    table.add_cell(2,0, width = 0.1, height = 0.3, text='Classifier', loc='center', fontproperties = fontproperties, facecolor = 'white')
    table[2,0].set_text_props(rotation = 'vertical')
    table[2,0].visible_edges = 'LTR'
    table.add_cell(2,1, width = 0.2, height = 0.3, text=f'Positive\n\n{(conf_mat[0][0]+conf_mat[0][1])/tot:.2%}', loc='center', fontproperties = fontproperties, facecolor = 'white')
    table.add_cell(2,2, width = 0.3, height = 0.3, text=f'{conf_mat[0][0]/orig_pos:.2%}', loc='center', fontproperties = fontproperties, facecolor = cell_color[1])
    table[2,2].set_text_props(c = text_color[0][0])
    table.add_cell(2,3, width = 0.3, height = 0.3, text=f'{conf_mat[0][1]/orig_neg:.2%}', loc='center', fontproperties = fontproperties, facecolor = cell_color[2])
    table[2,3].set_text_props(c = text_color[0][1])
    table.add_cell(2,4, width = 0.2, height = 0.3, text=f'Precision\n\n{precision:.2%}', loc='center', fontproperties = fontproperties, facecolor = 'white')

    table.add_cell(3,0, width = 0.1, height = 0.3, text='Output of', loc='center', fontproperties = fontproperties, facecolor = 'white')
    table[3,0].set_text_props(rotation = 'vertical')
    table[3,0].visible_edges = 'BLR'
    table.add_cell(3,1, width = 0.2, height = 0.3, text=f'Negative\n\n{(conf_mat[1][0]+conf_mat[1][1])/tot:.2%}', loc='center', fontproperties = fontproperties, facecolor = 'white')
    table.add_cell(3,2, width = 0.3, height = 0.3, text=f'{conf_mat[1][0]/orig_pos:.2%}', loc='center', fontproperties = fontproperties, facecolor = cell_color[3])
    table[3,2].set_text_props(c = text_color[1][0])
    table.add_cell(3,3, width = 0.3, height = 0.3, text=f'{conf_mat[1][1]/orig_neg:.2%}', loc='center', fontproperties = fontproperties, facecolor = cell_color[4])
    table[3,3].set_text_props(c = text_color[1][1])
    table.add_cell(3,4, width = 0.2, height = 0.3)

    table.add_cell(4,2, width = 0.3, height = 0.2, text=f'Sensitivity\n\n{sensitivity:.2%}', loc='center', fontproperties = fontproperties, facecolor = 'white')
    table.add_cell(4,3, width = 0.3, height = 0.2, text=f'Specificity\n\n{specificity:.2%}', loc='center', fontproperties = fontproperties, facecolor = 'white')
    table.add_cell(4,4, width = 0.2, height = 0.2, text=f'Acc: {accuracy:.2%}\n\nF1: {F1score:.2}', loc='center', fontproperties = fontproperties, facecolor = 'white')
    
    ax.add_table(table)

    return fig, ax

def gaussian(x,a,mean,sigma):
    return a*np.exp(-((x-mean)**2/(sigma**2))/2)

def hp_mode(mu1,mu2):
    if mu1 < mu2:
        return 'less'
    else:
        return 'greater'
    
def check_var(sig1,sig2):
    if sig1==sig2:
        return True
    else:
        return False
        
def plot_histo_gaus_stat(dist1, label1, dist2, label2):
    fig, ax = plt.subplots(figsize=(16,9))

    bin_vals1, bins1, _ = ax.hist(dist1, bins='auto', alpha = 0.5, color='red', label = label1)
    bin_vals2, bins2, _ = ax.hist(dist2, bins='auto', alpha = 0.5, color='blue', label = label2)

    mask1 = np.where(bin_vals1 != 0)
    bin_centers1 = (bins1[:-1] + bins1[1:])/2
    i_max1 = np.argmax(bin_vals1)
    par1 = [bin_vals1[i_max1], np.mean(dist1), np.std(dist1)]

    mask2 = np.where(bin_vals2 != 0)
    bin_centers2 = (bins2[:-1] + bins2[1:])/2
    i_max2 = np.argmax(bin_vals2)
    par2 = [bin_vals2[i_max2], np.mean(dist2), np.std(dist2)]
    
    try:
        popt1, pcov1 = curve_fit(gaussian, bin_centers1[mask1], bin_vals1[mask1], par1, maxfev=10000)
        popt2, pcov2 = curve_fit(gaussian, bin_centers2[mask2], bin_vals2[mask2], par2, maxfev=10000)

        x = np.linspace(np.min(np.concatenate((bins1, bins2))), np.max(np.concatenate((bins1,bins2))), 1000)
        ax.plot(x, gaussian(x,*popt1), 'r--', label='Gaussian Fit: A = {:.2f}, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(popt1[0], popt1[1], popt1[2]))
        ax.plot(x, gaussian(x,*popt2), 'b--', label='Gaussian Fit: A = {:.2f}, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(popt2[0], popt2[1], popt2[2]))

        stat_res = ttest_ind(dist1, dist2 ,equal_var=check_var(popt1[2], popt2[2]), alternative=hp_mode(popt1[1], popt2[1]))
    except:
        x = np.linspace(np.min(np.concatenate((bins1, bins2))), np.max(np.concatenate((bins1,bins2))), 1000)
        ax.plot(x, gaussian(x,*par1), 'r--', label='A = {:.2f}, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(par1[0], par1[1], par1[2]))
        ax.plot(x, gaussian(x,*par2), 'b--', label='A = {:.2f}, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(par2[0], par2[1], par2[2]))

        stat_res = ttest_ind(dist1, dist2 ,equal_var=check_var(np.var(dist1), np.var(dist2)), alternative=hp_mode(np.mean(dist1), np.mean(dist2)))

    ax.plot([],[], marker= None, linestyle='None', label='t-stat: {:.2f}, p-value: {:.2f}'.format(stat_res.statistic, stat_res.pvalue))

    ax.legend(loc='best')

    return fig, ax, stat_res

def torch_eig(mat, var_type):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Torch Device: ', device.type)

    torch_mat = torch.from_numpy(mat).to(device, dtype=var_type)

    e_val, e_vec = torch.linalg.eig(torch_mat)

    e_val = e_val.cpu().numpy()
    e_vec = e_vec.cpu().numpy()

    torch.cuda.empty_cache()

    return e_val, e_vec

def heatmap_plotter(ax, x, y, array, text_format, title, x_label, y_label, norm, cmap = cm.viridis):
    colormesh = ax.pcolormesh(x, y, array, norm=norm, cmap=cmap)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if text_format != None:
        rows,cols = array.shape
        for row in range(rows):
            for col in range(cols):
                if norm(array[row][col]) < 0.5:
                    ax.text(x[col], y[row], text_format.format(array[row,col]), ha='center', va='center', color=cmap(0.99), fontsize='xx-small')
                else:
                    ax.text(x[col], y[row], text_format.format(array[row,col]), ha='center', va='center', color=cmap(0.), fontsize='xx-small')

    return colormesh

def squared_error(pred, target):
    return (pred - target)**2

def step_squared_error(pred, target):
    return (pred - target) * pred*(1-pred)

def perceptron_loss(pred, target):
    return abs(pred - target)

def step_perceptron_loss(pred, target):
    return pred - target

def heavside(x):
    return np.where(x>0.5, 1, 0)

class neuron:

    def __init__(self, labneg, labpos, max_epochs, learning_rate, tol = 1e-3, loss = 'squaredError', random_state = None):
        self.labneg = labneg
        self.labpos = labpos
        self.epochs = max_epochs
        self.lr = learning_rate
        self.tol = tol
        random.seed(random_state)

        self.conv_epoch = -1

        if loss == 'squaredError':
            self.lossFunc = squared_error
            self.d_cost = step_squared_error
            self.activ = expit
        
        elif loss == 'perceptron':
            self.lossFunc = perceptron_loss
            self.d_cost = step_perceptron_loss
            self.activ = heavside

        else:
            print(f'{loss} loss function unknown')

    def fit(self, patterns, target):
        self.n_feat = len(patterns[0])
        self.train_patt = patterns
        self.train_lab = np.where(target == self.labpos, 1., 0.)
        self.weights = np.empty(self.n_feat)
        for i in range(self.n_feat):
            self.weights[i] = 2*random.random() - 1
        
        self.loss_list = []
        self.conv_counter = 0

        for i in range(self.epochs):
            # Forward Propagation
            self.weighted_sum = self.weights.dot(self.train_patt.transpose())
            self.temp_pred = self.activ(self.weighted_sum)
            self.epoch_loss = 0.

            for k, tpred in enumerate(self.temp_pred):
                # Error Computation
                self.loss = self.lossFunc(tpred, self.train_lab[k])
                self.epoch_loss += self.loss

                # Back Propagation
                self.step = self.d_cost(tpred, self.train_lab[k])
                for j in range(self.n_feat):
                    self.weights[j] -= self.lr * self.step * self.train_patt[k, j]

            # Convergence Check
            if self.conv_counter < 5 and i>0:
                if np.abs(self.epoch_loss - self.loss_list[i-1]) < self.tol:
                    self.conv_counter += 1
                else:
                    self.conv_counter = 0
            
            if self.conv_counter == 5:
                self.conv_epoch = i

            self.loss_list.append(self.epoch_loss)

    def predict(self, patterns):
        return np.where(self.activ(self.weights.dot(patterns.transpose())) > 0.5, self.labpos, self.labneg) 

