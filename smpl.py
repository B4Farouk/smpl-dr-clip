"""
    This class regroups our methods and classes for a differentiable SMPL
"""

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from pytorch3d.structures import Meshes

class SMPL:
    __DEFAULT_MODEL = SMPL_Layer(model_root='/content/project')
    def __init__(self, model, device):
        self.__device = device if device is not None else torch.device("cpu")
        self.__model = model if model is not None else SMPL.__DEFAULT_MODEL
        
    def to(self, device):
        assert device is not None
        self.__model.to(device)
        
    # theta is the pose parameter of shape (1,72) 
    # beta is the shape parameter of shape (1,10)
    def meshes(self, betas, thetas):
        # move the model to the device
        self.__model.to(self.__device)
        # create the vertices and joints of the mesh
        vertices, joints = self.__model.forward(th_pose_axisang=thetas, th_betas=betas, th_trans=None)
        # create the mesh
        mesh = Meshes(vertices, self.__model.th_faces[None, :])
        return mesh
        