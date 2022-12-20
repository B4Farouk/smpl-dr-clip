"""
    This module regroups our functions and classes used to optimize SimpledCLIP
"""
# torch imports
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn.functional import cosine_similarity

import pandas as pd

def init_weights(device):
    """
    Initializes the pose and shape tensor parameters for optimization.

    Args:
        device: the device on which to create those tensors.

    Returns:
        pose tensor, shape tensor
    """
    pose  = torch.zeros((1, 72), requires_grad=True, device=device) 
    shape = torch.ones((1, 10), requires_grad=True, device=device)
    return pose, shape

def init_random_weights(device):
    """
    Randomly initializes the pose and shape tensor parameters for optimization.

    Args:
        device: the device on which to create those tensors.

    Returns:
        pose tensor, shape tensor
    """
    pose  = torch.rand((1, 72), requires_grad=True, device=device)-0.5 
    shape = torch.ones((1, 10), requires_grad=True, device=device)
    return pose, shape

class TrackerConfig:
    """
    A class that encapsulates the loss tracking configuration for our optimizer.
    """
    def __init__(self, **kwargs):
        # pose
        self.track_pose = kwargs.get("track_pose", True)
        self.pose_freq  = kwargs.get("pose_freq", 10)
        # shape
        self.track_shape = kwargs.get("track_shape", True)
        self.shape_freq  = kwargs.get("shape_freq", 10)
        # loss
        self.track_loss = kwargs.get("track_loss", True)
        self.loss_freq  = kwargs.get("loss_freq", 10)

class OptimConfig:
    """
    A class that encapsulates the optimization configuration for our optimizer.
    """
    __DEFAULT_MULTI_IMAGE_LOSS_MODE = "average-loss-on-embeddings"
    __DEFAULT_LOSS_FN = lambda u, v: 1 - cosine_similarity(u, v, dim=1, eps=1e-8)
    
    def __init__(self, **kwargs):
        # optimizer params
        self.lr = kwargs.get("lr", 1e-3)
        self.betas = kwargs.get("betas", (0.9, 0.999))
        # LR scheduler params
        self.use_sch = kwargs.get("use_sch", False)
        self.sch_freq = kwargs.get("sch_freq", 1)
        self.sch_factor = kwargs.get("sch_factor", 0.5)
        self.sch_min_lr = kwargs.get("sch_min_lr", 5*1e-5)
        self.sch_threshold = kwargs.get("sch_threshold", 1e-3)
        self.sch_patience = kwargs.get("sch_patience", 10)
        self.sch_cooldown = kwargs.get("sch_cooldown", 100)
        self.sch_verbose = kwargs.get("sch_verbose", False) 
        # loss params
        self.loss_mode = kwargs.get("loss_mode", OptimConfig.__DEFAULT_MULTI_IMAGE_LOSS_MODE)
        self.loss_fn = kwargs.get("loss_fn", OptimConfig.__DEFAULT_LOSS_FN)

class OptimEnv:
    """
    A class that implements the optimization environment.
    """
    def __init__(self, model, weights, config: OptimConfig):
        # model
        self.__model = model
        # optim config
        self.__config = config
        # optimizer
        self.__optimizer = Adam(params=weights, lr=self.__config.lr, betas=self.__config.betas)
        # scheduler
        if self.__config.use_sch:
            self.__lr_scheduler = ReduceLROnPlateau(
                optimizer=self.__optimizer, 
                mode="min",
                factor=self.__config.sch_factor,
                min_lr=self.__config.sch_min_lr,
                threshold=self.__config.sch_threshold,
                patience=self.__config.sch_patience,
                cooldown=self.__config.sch_cooldown,
                verbose=self.__config.sch_verbose)
        
    def _forward(self, pose, shape):
        """
        implements the forward pass
        """
        imgs_embs, pmt_emb = self.__model(pose, shape)
        
        if self.__config.loss_mode == "loss-on-average-embedding":
            loss = self.__config.loss_fn(imgs_embs.mean(axis=0, keepdims=True), pmt_emb)
        elif self.__config.loss_mode == "average-loss-on-embeddings":
            loss = self.__config.loss_fn(imgs_embs, pmt_emb).mean()
        else:
            raise ValueError("incorrect loss mode")
        
        return loss
        
    def _backward(self, loss):
        """
        impelements the backward pass
        """
        loss.backward(retain_graph=True)
        
    def _opti_step(self):
        """
        implements the optimizer's update
        """
        self.__optimizer.step()
        self.__optimizer.zero_grad()
        
    @staticmethod
    def _coorddesc_gradmask(pose):
        """
        Returns a gradient mask to implement coordinate descent, by masking the gradient of all but one single parameter in the pose tensor.

        Args:
            pose: the pose parameter.

        Returns:
            a gradient mask to implement coordinate descent.
        """
        gradmask = torch.zeros_like(pose)
        joint_id = torch.randint(low=0, high=71, size=(1,1)).item()
        gradmask[:, joint_id*3:(joint_id+1)*3] = 1
        return gradmask
        
    def optimize(self, pose, shape, n_passes=1000, coorddesc=False, gradmask=None, trackerconfig: TrackerConfig = None):
        """
        Optimizes SimpledCLIP.

        Args:
            pose: initial pose tensor
            shape: initial shape tensor
            n_passes: the number of passes. Defaults to 1000.
            coorddesc: whether to do coordinate descent. Defaults to False.
            gradmask: an explicit gradient mask used to nullify the updates and thus freeze of some pose parameters. Defaults to None.
            trackerconfig: a TrackerConfig instance. Defaults to None.

        Returns:
            A dictionnary summarizing the result and tracked objects, if any, during optimization.
        """
        # tracker dataframes
        if trackerconfig.track_loss:
            intermediate_losses = pd.DataFrame(columns=["pass", "loss"])
        if trackerconfig.track_pose:
            intermediate_poses = pd.DataFrame(columns=["pass", "pose"])
        if trackerconfig.track_shape:
            intermediate_shapes = pd.DataFrame(columns=["pass", "shape"])
        
        # optimizaiton loop
        for n in range(1, n_passes+1):
            # foward + backward passes
            loss = self._forward(pose, shape)
            self._backward(loss)
            # nullify the gradient of unoptimized coordinates
            if gradmask is not None:
                pose.grad *= gradmask
            elif coorddesc:
                pose.grad *= OptimEnv._coorddesc_gradmask(pose)
            # optimizer step
            self._opti_step()
            # LR scheduler step
            if self.__config.use_sch and (n % self.__config.sch_freq == 0):
                self.__lr_scheduler.step(metrics=loss)
            # loss tracking
            if trackerconfig.track_loss and (n % trackerconfig.loss_freq == 0):
                intermediate_losses.loc[len(intermediate_losses)] = {"pass": n, "loss": loss.item()}
            # pose tracking
            if trackerconfig.track_pose and (n % trackerconfig.pose_freq == 0):
                intermediate_poses.loc[len(intermediate_poses)] = {"pass": n, "pose": pose.cpu().detach()}
            # shape tracking
            if trackerconfig.track_shape and n % trackerconfig.shape_freq == 0:
                intermediate_shapes.loc[len(intermediate_shapes)] = {"pass": n, "shape": shape.cpu().detach()}
        
        # if tracker_config is None: result["tracked"] is None
        #   if not(track_loss): result["tracked"]["losses"] is None,
        #   similarly for other entries in result["tracked"] for example for result["tracked"]["pose"]
        result = {
            "params": {"pose": pose, "shape": shape},
            "tracked": {
                "losses": intermediate_losses if trackerconfig.track_loss else None, 
                "poses": intermediate_poses if trackerconfig.track_pose else None,
                "shapes": intermediate_shapes if trackerconfig.track_shape else None
                } if trackerconfig is not None else None
        }
        
        return result
