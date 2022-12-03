"""
    This module regroups our functions and classes used to train the model
"""
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

def init_weights(device):
    pose  = torch.zeros((1, 72), requires_grad=True, device=device) # theta
    shape = torch.ones((1, 10), requires_grad=True, device=device) # beta
    return pose, shape

class OptimEnv:
    __DEFAULT_SIM_FN  = nn.CosineSimilarity(dim=1, eps=1e-8)
    __DEFAULT_LOSS_FN = lambda img_emb, prompt_emb: 1 - OptimEnv.__DEFAULT_SIM_FN(img_emb, prompt_emb)
    
    def __init__(self, model, params, lr=1e-3, betas=(0.9, 0.999)):
        self.__model = model
        self.__params = params
        self.__optimizer = optim.Adam(params=self.__params, lr=lr, betas=betas)#, amsgrad=True)
        self.__loss_fn = OptimEnv.__DEFAULT_LOSS_FN
        
    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer
        
    def set_loss_fn(self, loss_fn):
        self.__loss_fn = loss_fn
    
    def forward(self, pose, shape):
        image_embedding, prompt_embedding = self.__model(pose, shape)
        loss = self.__loss_fn(image_embedding, prompt_embedding)
        return loss
        
    def backward(self, loss):
        self.__optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.__optimizer.step()
    
    def optimize(self, pose, shape, n_passes=1000, track_loss=True, every_npasses=50):
        if track_loss:
            tracked_losses = np.array([])

        for n in range(n_passes):
            loss = self.forward(pose, shape)
            self.backward(loss)
            
            if track_loss and n % every_npasses == 0:
                tracked_losses = np.append(tracked_losses, loss.item())
                    
        return pose, shape, tracked_losses