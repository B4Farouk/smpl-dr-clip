"""
    This module regroups our functions and classes used to train the model
"""
# torch imports
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn.functional import cosine_similarity

import pandas as pd

def init_weights(device):
    pose  = torch.zeros((1, 72), requires_grad=True, device=device) # theta
    shape = torch.ones((1, 10), requires_grad=True, device=device) # beta
    return pose, shape

class OptimEnv:
    __DEFAULT_MULTI_IMAGE_LOSS_MODE = "average-loss-on-embeddings"
    
    def __init__(self, model, weights, activate_lr_sch, config={}):
        # model
        self.__model = model
        
        # loss function
        self.__loss_fn = lambda u, v,: 1 - cosine_similarity(u, v, dim=1, eps=1e-8)
        
        # optimizer
        lr = config.get("lr", 1e-3)
        betas = config.get("betas", (0.9, 0.999))
        self.__optimizer = Adam(params=weights, lr=lr, betas=betas)
        
        # loss mode
        self.__loss_mode = config.get("loss_mode", OptimEnv.__DEFAULT_MULTI_IMAGE_LOSS_MODE)
        
        # LR scheduler
        self.__activate_lr_sch = activate_lr_sch
        
        sch_factor = config.get("sch_factor", 0.5)
        sch_min_lr = config.get("sch_min_lr", 5*1e-5)
        
        sch_threshold = config.get("sch_threshold", 1e-3)
        sch_patience = config.get("sch_patience", 10)
        sch_cooldown = config.get("sch_cooldown", 0)
        
        sch_verbose = config.get("sch_verbose", False)
        
        if activate_lr_sch:
            self.__lr_scheduler = ReduceLROnPlateau(
                optimizer=self.__optimizer, 
                mode="min",
                factor=sch_factor,
                min_lr=sch_min_lr,
                threshold=sch_threshold,
                
                patience=sch_patience,
                cooldown=sch_cooldown,
                
                verbose=sch_verbose)
        
    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer
        
    def set_loss_fn(self, loss_fn):
        self.__loss_fn = loss_fn
    
    def set_lr_scheduler(self, scheduler):
        self.__lr_scheduler = scheduler
    
    def forward(self, pose, shape):
        imgs_embs, pmt_emb = self.__model(pose, shape)
        
        if self.__loss_mode == "loss-of-average-embedding":
            loss = self.__loss_fn(imgs_embs.mean(axis=0, keepdims=True), pmt_emb)
        elif self.__loss_mode == "average-loss-on-embeddings":
            loss = self.__loss_fn(imgs_embs, pmt_emb).mean()
        else:
            raise ValueError("incorrect loss mode")
        
        return loss
        
    def backward(self, loss):
        self.__optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.__optimizer.step()
    
    def optimize(self, pose, shape, n_passes=1000, tracker_config=None):
        # get tracker settings
        pose_tracker_config = tracker_config.get("pose", None)
        shape_tracker_config = tracker_config.get("shape", None)
        loss_tracker_config = tracker_config.get("loss", None)
        
        track_loss = loss_tracker_config is not None
        if track_loss:
            loss_interleaving = loss_tracker_config.get("interleaving", 50)
            intermediate_losses = pd.DataFrame(columns=["pass", "loss"])
            
        track_pose = pose_tracker_config is not None
        if track_pose:
            pose_interleaving = pose_tracker_config.get("interleaving", 100)
            intermediate_poses = pd.DataFrame(columns=["pass", "pose"])
        
        track_shape = shape_tracker_config is not None
        if track_shape:
            shape_interleaving = pose_tracker_config.get("interleaving", 100)
            intermediate_shapes = pd.DataFrame(columns=["pass", "shape"])
        
        # optimizaiton loop
        for n in range(1, n_passes+1):
            if(n%100==0):
                print('number of passes is: '+ str(n_passes))
            # optimization steps: forward pass + zero_grad + backward pass + optimizer step
            loss = self.forward(pose, shape)
            self.backward(loss)
            if self.__activate_lr_sch:
                # LR scheduler update after each iteration, seems to be the right to do with the current schedueler: ReduceLROnPlateau  
                self.__lr_scheduler.step(metrics=loss)
                          
            # loss tracking
            if track_loss and n % loss_interleaving == 0:
                intermediate_losses.loc[len(intermediate_losses)] = {"pass": n, "loss": loss.item()}
            # pose tracking
            if track_pose and n % pose_interleaving == 0:
                intermediate_poses.loc[len(intermediate_poses)] = {"pass": n, "pose": pose.cpu().detach()}
            # shape tracking
            if track_shape and n % shape_interleaving == 0:
                intermediate_shapes.loc[len(intermediate_shapes)] = {"pass": n, "shape": shape.cpu().detach()}
        
        # if tracker_config is None: result["tracked"] is None
        #   if not(track_loss): result["tracked"]["losses"] is None,
        #   similarly for other entries in result["tracked"] for example for result["tracked"]["pose"]
        result = {
            "params": {"pose": pose, "shape": shape},
            "tracked": {
                "losses": intermediate_losses if track_loss else None, 
                "poses": intermediate_poses if track_pose else None,
                "shapes": intermediate_shapes if track_shape else None
                } if tracker_config is not None else None
        }
        
        return result
