"""
    This module regroups our functions and classes used to train the model
"""
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

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
    
    def optimize(self, pose, shape, n_passes=1000, tracking_settings=None):
        
        # tracking the loss during optimization
        track_loss = tracking_settings is not None
        if track_loss:
            interleaving = tracking_settings.get("interleaving", 50)
            losses = pd.DataFrame(columns=["pass_num", "loss"])

        # optimizaiton loop
        for n in range(1, n_passes+1):
            # optimization steps: forward pass + zero_grad + backward pass + optimizer step
            loss = self.forward(pose, shape)
            self.backward(loss)
            # loss tracking every interleaving number of passes
            if track_loss and n % interleaving == 0:
                losses.loc[len(losses)] = {"pass_num": n, "loss": loss.item()}
            
        # different return signature depending on whether we track the loss
        if track_loss:
            return (pose, shape), losses
        return pose, shape