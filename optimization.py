"""
    This module regroups our functions and classes used to train the model
"""
import torch.nn as nn
import torch.optim as optim

def cos_dist(image_embedding, prompt_embedding):
        # Compute the cosine similarity as a tensor
        cos_sim = image_embedding.cpu() @ prompt_embedding.cpu().T
        return  1 - cos_sim

class OptimEnv:
    __DEFAULT_LOSS_FN = cos_dist
    def __init__(self, model, params, lr=1, betas=(0.9, 0.999)):
        self.__model = model
        self.__params = params
        self.__optimizer = optim.Adam(params=self.__params, lr=lr, betas=betas, amsgrad=True)
        self.__loss_fn = OptimEnv.__DEFAULT_LOSS_FN
        
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
    
    def optimize(self, pose, shape, prompt, n_passes=1000, verbose=True):
        for n in range(n_passes):
            loss = self.forward(pose, shape, prompt)
            self.backward(loss)
            
            if verbose:
                if n % 100 == 0:
                    print("iteration %d: loss = %.5f"%(n, loss.item()))
                    
        return pose, shape