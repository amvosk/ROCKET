# import math
# import pandas as pd
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = arguments.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse

import numpy as np
import torch
# import torch.nn as nn
# import einops
import h5py
from tqdm import tqdm

from util import load_dataset
from estimators import SingleTrialEstimate, MCDropoutEstimate, EnsembleEstimate, FFTEstimate

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--low", type = int, default = 0)
parser.add_argument("-m", "--high", type = int, default = 85)
# parser.add_argument("-g", "--gpu", type = str, default = '2')

arguments = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = arguments.gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

path_dataset = '../Univariate_arff/'
path_results = '../results/'
dataset85 = '../dataset85.txt'

with open(dataset85, 'r') as f:
    dataset_names = list(map(str.strip, f.readlines()))
dataset_names = dataset_names[arguments.low:arguments.high]
    
n_datasets = len(dataset_names)
n_kernels = 10000
random_seed = 0
n_ece_bins = 10

n_samples = 100
mc_dropout = 0.01
n_samples_mc = 1000
n_random_features = 10
n_samples = 100
# estimators_names = ['ste', 'mcde', 'ee', 'ffteO', 'ffteE', 'ffteF', 'ffteU']
estimators_names = ['ste', 'mcde', 'ee', 'ffteU']

for dataset_name in tqdm(dataset_names):
    
    
    X_training, Y_training, X_test, Y_test, n_classes, classes_old = load_dataset(path_dataset, dataset_name)
    ts_length = X_training.shape[1]

    for estimators_name in estimators_names:
        path_save = path_results + dataset_name + '_' + estimators_name + '.h5'
        if os.path.isfile(path_save):
            continue
        if dataset_name in ['ElectricDevices', 'StarLightCurves', 'UWaveGestureLibraryAll']:
            continue
        
        if estimators_name == 'ste':
            est = SingleTrialEstimate(n_kernels, ts_length, n_classes, n_ece_bins, random_seed)
        elif estimators_name == 'mcde':
            est = MCDropoutEstimate(n_kernels, ts_length, n_classes, n_ece_bins, random_seed, mc_dropout, n_samples_mc)
        elif estimators_name == 'ee':
            est = EnsembleEstimate(n_kernels, ts_length, n_classes, n_ece_bins, random_seed, n_random_features)
        elif estimators_name == 'ffteO':
            est = FFTEstimate(n_kernels, ts_length, n_classes, n_ece_bins, random_seed, n_samples, 'odd')
        elif estimators_name == 'ffteE':
            est = FFTEstimate(n_kernels, ts_length, n_classes, n_ece_bins, random_seed, n_samples, 'even')
        elif estimators_name == 'ffteF':
            est = FFTEstimate(n_kernels, ts_length, n_classes, n_ece_bins, random_seed, n_samples, 'full')
        elif estimators_name == 'ffteU':
            est = FFTEstimate(n_kernels, ts_length, n_classes, n_ece_bins, random_seed, n_samples, 'unif')

        # t1 = time.time()
        # path_save = path_results + dataset_name + '_' + estimators_name + '.h5'
        est.fit(X_training, Y_training)
        result = est.score(X_test, Y_test)
        # print((time.time() - t1)/60, estimators_name, result)

        with h5py.File(path_save, 'w') as file:
            for k, v in result.items():
                file.attrs[k] = v