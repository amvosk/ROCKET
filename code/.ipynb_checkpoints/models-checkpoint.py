import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import math
import numpy as np
import torch
import torch.nn as nn
import einops
from tqdm import tqdm

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

class RandomFeatures(nn.Module):
    kernel_length = torch.tensor([7, 9, 11], dtype=torch.float)
    weight = None
    bias = None
    dilation = None
    padding = None
    
    def __init__(self, n_kernels, ts_length, random_seed=0):
        super(RandomFeatures, self).__init__()
        
        self.n_kernels = n_kernels
        self.ts_length = ts_length
        self.random_seed = random_seed * n_kernels
        self.create_kernels(self.random_seed)
        
    def create_kernels(self, random_seed=0):
        torch.manual_seed(random_seed)
        kernel_length_indices = torch.multinomial(self.kernel_length, self.n_kernels, replacement=True)
        kernel_lengths = self.kernel_length[kernel_length_indices].to(torch.long)
        weight_distributions_ = [torch.normal(mean=0, std=1, size=(kernel_length,), dtype=torch.double) for kernel_length in kernel_lengths]
        weight_distributions = [weight_distribution - weight_distribution.mean() for weight_distribution in weight_distributions_]
        self.weight = [weight_distribution.unsqueeze(0).unsqueeze(0) for weight_distribution in weight_distributions]
        
        self.bias = torch.rand(size=(self.n_kernels,), dtype=torch.double).reshape(-1,1) * 2 - 1
        A = math.log(self.ts_length-1) - torch.log(kernel_lengths-1)
        s = torch.rand(size=(self.n_kernels,)) * A
        self.dilation = torch.floor(2**s)
        self.padding = ((self.dilation * (kernel_lengths-1) / 2) * torch.randint(2, size=(self.n_kernels,)))
    
    
    def forward(self, x):
        batch_size = x.size(0)
        x = einops.rearrange(x, 'b t -> b 1 t')
        features = torch.empty(size=(batch_size, self.n_kernels, 2))
        
        for i in range(self.n_kernels):
            ts_convolved = nn.functional.conv1d(
                x, weight=self.weight[i], bias=self.bias[i], 
                padding=int(self.padding[i].item()), dilation=int(self.dilation[i].item())
            )
            ts_convolved = ts_convolved.squeeze(1)

            ts_convolved_max = torch.max(ts_convolved, dim=1).values
            features[:,i,0] = ts_convolved_max

            ts_convolved_ppv = torch.mean((ts_convolved > 0).to(torch.float), dim=1)
            features[:,i,1] = ts_convolved_ppv
        
        features = einops.rearrange(features, 'b s f -> b (s f)')
        return features
    
class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes, mc_dropout=0.3, random_seed=0):
        super(LogisticRegression, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.mc_dropout = mc_dropout
        self.random_seed = random_seed
        
        torch.manual_seed(self.random_seed)
        self.linear = nn.Linear(n_features, n_classes)
        self.input_dropout = nn.Dropout(p=self.mc_dropout)
    
    def update_dropout(self, mc_dropout=0.3):
        self.input_dropout = nn.Dropout(p=mc_dropout)
    
    def forward(self, x, use_dropout=False):
        if use_dropout:
            x = self.input_dropout(x)
        logits = self.linear(x)
        return logits
    
class Trainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1)
        self.criterion = nn.CrossEntropyLoss()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.train_loss = []

    def fit(self, x, y, nepoch=1, lr=1e-3, tol=None, counter_thr = 10):
        optimizer = torch.optim.Adam(
            [{'params':param, 'lr':lr} for param in self.model.parameters()]
        )
        counter = 0
        # l = range(nepoch)
        # if x.shape[0] > 5000:
        #     l = tqdm(l)
        
        for epoch in range(nepoch):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            optimizer.step()
            self.train_loss.append(loss.detach().cpu().numpy().item())
            if tol is not None and epoch > 1:
                if abs(self.train_loss[-1] - self.train_loss[-2]) <= tol:
                    counter += 1
                else:
                    counter = 0
            if counter >= counter_thr:
                break
            
    def predict_logits(self, x):
        self.model.eval()
        # logits_unscaled = torch.empty(x.shape[0], self.model.n_classes)
        # if x.shape[0] > 4000:
        #     split = x.shape[0] // 2
        #     logits_unscaled[:a] = self.model(x[:a])
        #     logits_unscaled[a:] = self.model(x[a:])
        # else:
        logits_unscaled = self.model(x)
        logits = self.logsoftmax(logits_unscaled)
        return logits
    
    def predict_mc_logits(self, x, n_samples_mc=100):
        batch_size = x.size(0)
        self.model.train()
        logits_unscaled = torch.empty(size=(batch_size, self.model.n_classes, n_samples_mc))
        for i in range(n_samples_mc):
            # if x.shape[0] > 4000:
            #     split = x.shape[0] // 2
            #     logits_unscaled[:a,:,i] = self.model(x[:a], use_dropout=True).detach().clone()
            #     logits_unscaled[a:,:,i] = self.model(x[a:], use_dropout=True).detach().clone()
            # else:
            logits_unscaled_ = self.model(x, use_dropout=True)
            logits_unscaled[:,:,i] = logits_unscaled_.detach().clone()
        logits_sum = self.logsoftmax(logits_unscaled)
        logits = torch.logsumexp(logits_sum, dim=2) - math.log(n_samples_mc)
        return logits