import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

def accuracy(logits, targets):
    # probs = np.exp(logits.detach().cpu().numpy())
    probs = torch.exp(logits)
    correspondence = (torch.argmax(probs, axis=1) == targets).to(torch.float)
    return correspondence.mean().detach().cpu().numpy().item()

def negative_loglikelihood(logits, targets):
    nll = nn.NLLLoss()
    return nll(logits, targets).detach().cpu().numpy().item()

def brier_score(logits, targets, n_classes):
    probs = torch.exp(logits)
    targets_one_hot = nn.functional.one_hot(targets, num_classes=n_classes)
    square_errors = (probs - targets_one_hot)**2
    return square_errors.mean().detach().cpu().numpy().item()

def expected_calibration_error(logits, targets, n_bins=10):
    bins_low = torch.arange(n_bins) / n_bins
    bins_high = (torch.arange(n_bins) + 1) / n_bins

    probs = torch.exp(logits)
    p_hat = torch.max(probs, dim=1).values
    targets_hat = torch.argmax(probs, dim=1)
    accuracy = (targets_hat == targets)
    
    calibration_errors = []
    for i in range(n_bins):
        mask = ((p_hat >= bins_low[i] if i==0 else p_hat > bins_low[i]) * (p_hat <= bins_high[i]))
        if torch.any(mask):
            bin_size = torch.sum(mask.to(torch.float)).item()
            p_hat_masked = torch.mean(p_hat[mask].to(torch.float)).item()
            accuracy_masked = torch.mean(accuracy[mask].to(torch.float)).item()
            calibration_error = abs(accuracy_masked - p_hat_masked)
            calibration_errors.append(calibration_error * bin_size / targets.size(0))
    return np.sum(calibration_errors)