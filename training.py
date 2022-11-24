"""
    This module regroups our functions and classes used to train the model
"""
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, pose, shape):
        assert len(pose) == 72 and len(shape) == 10
        
        self.__model = model
        self.__params = [pose, shape]
        self.__optimizer = optim.Adam(params=self.__params, lr=1, amsgrad=True)
        self.__loss_fn = nn.MSELoss()
        
    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer
        
    def set_loss_fn(self, loss_fn):
        self.__loss_fn = loss_fn
    
    def forward(self, pose, shape, prompt):
        image_score, prompt_score = self.__model(pose, shape, prompt)
        loss = self.__loss_fn(image_score, prompt_score)
        return loss
        
    def backward(self, loss):
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
    