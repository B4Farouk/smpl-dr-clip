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
    
    def optimize(self, pose, shape, n_passes=1000, tracker_settings=None):
        # get tracker settings
        pose_tracker_settings = tracker_settings.get("pose", None)
        shape_tracker_settings = tracker_settings.get("shape", None)
        loss_tracker_settings = tracker_settings.get("loss", None)
        
        track_loss = loss_tracker_settings is not None
        if track_loss:
            loss_interleaving = loss_tracker_settings.get("interleaving", 50)
            intermediate_losses = pd.DataFrame(columns=["pass", "loss"])
            
        track_pose = pose_tracker_settings is not None
        if track_pose:
            pose_interleaving = pose_tracker_settings.get("interleaving", 100)
            intermediate_poses = pd.DataFrame(columns=["pass", "pose"])
        
        track_shape = shape_tracker_settings is not None
        if track_shape:
            shape_interleaving = pose_tracker_settings.get("interleaving", 100)
            intermediate_shapes = pd.DataFrame(columns=["pass", "shape"])
        
        # optimizaiton loop
        for n in range(1, n_passes+1):
            # optimization steps: forward pass + zero_grad + backward pass + optimizer step
            loss = self.forward(pose, shape)
            self.backward(loss)
            
            # loss tracking
            if track_loss and n % loss_interleaving == 0:
                intermediate_losses.loc[len(intermediate_losses)] = {"pass": n, "loss": loss.item()}
            # pose tracking
            if track_pose and n % pose_interleaving == 0:
                intermediate_poses.loc[len(intermediate_poses)] = {"pass": n, "pose": pose.cpu().detach()}
            # shape tracking
            if track_shape and n % shape_interleaving == 0:
                intermediate_shapes.loc[len(intermediate_shapes)] = {"pass": n, "pose": shape.cpu().detach()}
        
        # if tracker_settings is None: result["tracked"] is None
        #   if not(track_loss): result["tracked"]["losses"] is None,
        #   similarly for other entries in result["tracked"] for example for result["tracked"]["pose"]
        result = {
            "params": {"pose": pose, "shape": shape},
            "tracked": {
                "losses": intermediate_losses if track_loss else None, 
                "poses": intermediate_poses if track_pose else None
                } if tracker_settings is not None else None
        }
        
        return result