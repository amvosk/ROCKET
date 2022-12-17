import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import math
import numpy as np
import torch
import torch.nn as nn

from models import RandomFeatures, LogisticRegression, Trainer
from metrics import accuracy, negative_loglikelihood, brier_score, expected_calibration_error

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

class SingleTrialEstimate():
    def __init__(self, n_kernels, ts_length, n_classes, n_ece_bins, random_seed=0):
        self.n_ece_bins = n_ece_bins
        self.n_classes = n_classes
        self.random_seed = random_seed
        n_features = 2 * n_kernels
        self.rf = RandomFeatures(n_kernels, ts_length, random_seed)
        self.lr = LogisticRegression(n_features, n_classes, random_seed=random_seed)
        self.fitted = False
        self.trainer = None
        
    def fit(self, X, Y, nepoch=5000, lr=1e-4, tol=1e-4, counter_thr=300):
        X_ = torch.tensor(X, dtype=torch.double)
        Y = torch.tensor(Y, dtype=torch.long)

        features = self.rf(X_).detach().clone()
        del X_
        
        self.trainer = Trainer(self.lr)
        self.trainer.fit(features, Y, nepoch=nepoch, lr=lr, tol=tol, counter_thr=counter_thr)
        self.fitted = True
        
    def score(self, X, Y):
        assert self.fitted, 'model should be fitted'
        X_ = torch.tensor(X, dtype=torch.double)
        Y = torch.tensor(Y, dtype=torch.long)
        
        features = self.rf(X_).detach().clone()
        del X_
        
        logits = self.trainer.predict_logits(features)
        results = {}
        results['acc'] = accuracy(logits, Y)
        results['nll'] = negative_loglikelihood(logits, Y)
        results['bs'] = brier_score(logits, Y, self.n_classes)
        results['ece'] = expected_calibration_error(logits, Y, self.n_ece_bins)
        return results
    
class MCDropoutEstimate(SingleTrialEstimate):
    def __init__(self, n_kernels, ts_length, n_classes, n_ece_bins, random_seed=0,
                 mc_dropout=0.01, n_samples_mc=100):
        super(MCDropoutEstimate, self).__init__(n_kernels, ts_length, n_classes, n_ece_bins, random_seed)
        self.mc_dropout = mc_dropout
        self.n_samples_mc = n_samples_mc
        
    def score(self, X, Y):
        assert self.fitted, 'model should be fitted'
        X_ = torch.tensor(X, dtype=torch.double)
        Y = torch.tensor(Y, dtype=torch.long)

        features = self.rf(X_).detach().clone()
        del X_
        
        self.trainer.model.update_dropout(self.mc_dropout)
        logits = self.trainer.predict_mc_logits(features, self.n_samples_mc)
        results = {}
        results['acc'] = accuracy(logits, Y)
        results['nll'] = negative_loglikelihood(logits, Y)
        results['bs'] = brier_score(logits, Y, self.n_classes)
        results['ece'] = expected_calibration_error(logits, Y, self.n_ece_bins)
        return results
    
class EnsembleEstimate():
    def __init__(self, n_kernels, ts_length, n_classes, n_ece_bins, random_seed=0, 
                 n_random_features=10):
        self.n_ece_bins = n_ece_bins
        self.n_classes = n_classes
        self.random_seed = random_seed + 1
        self.n_random_features = n_random_features
        n_features = 2 * n_kernels
        self.rf = [RandomFeatures(n_kernels, ts_length, random_seed+i) for i in range(self.n_random_features)]
        self.lr = [LogisticRegression(n_features, n_classes, random_seed=random_seed+i) for i in range(self.n_random_features)]
        self.fitted = False
        self.trainer = None
        
    def fit(self, X, Y, nepoch=5000, lr=1e-4, tol=1e-4, counter_thr=300):

        Y = torch.tensor(Y, dtype=torch.long)
        self.trainer = [Trainer(lr) for lr in self.lr]
        for i in range(self.n_random_features):
            X_ = torch.tensor(X, dtype=torch.double)
            features = self.rf[i](X_).detach().clone()
            del X_
            self.trainer[i].fit(features, Y, nepoch=nepoch, lr=lr, tol=tol, counter_thr=counter_thr)
        self.fitted = True
        
    def score(self, X, Y):
        assert self.fitted, 'model should be fitted'
        X = torch.tensor(X, dtype=torch.double)
        Y = torch.tensor(Y, dtype=torch.long)
        
        logits_ensemble = torch.empty(size=(X.shape[0], self.n_classes, self.n_random_features))
        for i in range(self.n_random_features):
            X_ = torch.tensor(X, dtype=torch.double)
            features = self.rf[i](X).detach().clone()
            del X_
            logits_ = self.trainer[i].predict_logits(features)
            logits_ensemble[:,:,i] = logits_
        logits_ensemble = torch.nn.LogSoftmax(dim=1)(logits_ensemble)
        logits = torch.logsumexp(logits_ensemble, dim=2) - math.log(self.n_random_features)
            
        results = {}
        results['acc'] = accuracy(logits, Y)
        results['nll'] = negative_loglikelihood(logits, Y)
        results['bs'] = brier_score(logits, Y, self.n_classes)
        results['ece'] = expected_calibration_error(logits, Y, self.n_ece_bins)
        return results
    
class FFTEstimate(SingleTrialEstimate):
    def __init__(self, n_kernels, ts_length, n_classes, n_ece_bins, random_seed=0,
                 n_samples=10, style='full'):
        super(FFTEstimate, self).__init__(n_kernels, ts_length, n_classes, n_ece_bins, random_seed)
        self.n_samples = n_samples
        self.style = style

    def score(self, X, Y):
        assert self.fitted, 'model should be fitted'
        np.random.seed(self.random_seed)
        
        X = torch.tensor(X, dtype=torch.double)
        Y = torch.tensor(Y, dtype=torch.long)
        
        n_test = X.size(0)

        X_fft = torch.fft.fft(X, dim=1)
        X_fft_abs = np.abs(X_fft.detach().cpu().numpy())
        X_fft_angle = np.angle(X_fft.detach().cpu().numpy())

        X_fft_angle_sampled = []
        l = (X_fft_angle.shape[1]-1)//2
        
        if self.style == 'odd':
            amp = X_fft_abs[:,1:l+1][:,::2]
        elif self.style == 'even':
            amp = X_fft_abs[:,1:l+1][:,1::2]
        elif self.style == 'full':
            amp = X_fft_abs[:,1:l+1]
        elif self.style == 'unif':
            amp = np.ones_like(X_fft_abs)[:,1:l+1]
        p = amp / np.sum(amp, axis=1, keepdims=True)

        for i in range(self.n_samples):
            X_fft_angle_ = np.copy(X_fft_angle)
            for j in range(n_test):
     
                if self.style == 'odd':
                    index = np.random.choice(np.arange(1, l+1)[::2], size=1, p=p[j])
                elif self.style == 'even':
                    index = np.random.choice(np.arange(1, l+1)[1::2], size=1, p=p[j])
                elif self.style == 'full':
                    index = np.random.choice(np.arange(1, l+1), size=1, p=p[j])
                elif self.style == 'unif':
                    index = np.random.choice(np.arange(1, l+1), size=1, p=p[j])
                phase = (np.random.rand(1) - 0.5) * 2 * np.pi
                X_fft_angle_[j,index] = phase
                X_fft_angle_[j,-index] = - phase
            X_fft_angle_sampled.append(X_fft_angle_)

        X_fft_angle_sampled = np.stack(X_fft_angle_sampled)
        X_fft_sampled = X_fft_abs*np.exp(1j*X_fft_angle_sampled)

        # X_fft_sampled = torch.tensor(X_fft_abs*np.exp(1j*X_fft_angle_sampled))
        # X_sampled = torch.real(torch.fft.ifft(X_fft_sampled, dim=2))
        
        logits_sampled = torch.empty(n_test, self.n_classes, self.n_samples)
        for i in range(self.n_samples):
            X_fft_ = X_fft_sampled[i]
            # X_ = X_sampled[i]
            X_ = torch.real(torch.fft.ifft(torch.tensor(X_fft_), dim=1))
            features_ = self.rf(X_).detach().clone()
            del X_
            logits_ = self.trainer.predict_logits(features_).detach().clone()
            logits_sampled[:,:,i] = logits_

        logits_sampled = torch.nn.LogSoftmax(dim=1)(logits_sampled)
        logits = torch.logsumexp(logits_sampled, dim=2) - math.log(self.n_samples)
        
        results = {}
        results['acc'] = accuracy(logits, Y)
        results['nll'] = negative_loglikelihood(logits, Y)
        results['bs'] = brier_score(logits, Y, self.n_classes)
        results['ece'] = expected_calibration_error(logits, Y, self.n_ece_bins)
        return results
    
    
    
class RidgeClassifier():
    def __init__(self, n_kernels, ts_length, n_classes, random_seed=0):
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        
        self.n_classes = n_classes
        self.random_seed = random_seed
        n_features = 2 * n_kernels
        self.rf = RandomFeatures(n_kernels, ts_length, random_seed)
        
        self.rc = make_pipeline(StandardScaler(with_mean=False), RidgeClassifierCV(alphas = np.logspace(-3, 3, 10)))

        # self.rc = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        self.fitted = False
        
    def fit(self, X, Y, nepoch=5000, lr=1e-4, tol=1e-4, counter_thr=300):
        X_ = torch.tensor(X, dtype=torch.double)

        features = self.rf(X_).detach().clone().cpu().numpy()
        del X_
        
        self.rc.fit(features, Y)
        self.fitted = True
        
    def score(self, X, Y):
        assert self.fitted, 'model should be fitted'
        X_ = torch.tensor(X, dtype=torch.double)
     
        features = self.rf(X_).detach().clone().cpu().numpy()
        del X_
        
        results = {}
        results['acc'] = self.rc.score(features, Y)
        return results