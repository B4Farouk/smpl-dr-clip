"""
    This class regroups our methods and classes for a differentiable SMPL
"""

import torch

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesAtlas,
    TexturesUV,
    TexturesVertex
)
def mesh_from(vertices, faces, texture):
    return Meshes(
        verts=vertices, 
        faces=faces,
        textures=texture)

class SMPLwrapper:
    __DEFAULT_MODEL = SMPL_Layer(gender='female', model_root='project')
    
    def __init__(self, model, txmapping, batchsize, device):
        device = device if device is not None else torch.device("cpu")        
        
        self.__txmapping = txmapping
        
        self.__model = model if model is not None else SMPLwrapper.__DEFAULT_MODEL
        self.__model.to(device)
        
        self.__batchsize = 1 if batchsize is None or batchsize <= 0 else batchsize
         
    # theta is the pose parameter of shape (1,72) 
    # beta is the shape parameter of shape (1,10)
    def verts_and_faces(self, theta, beta):
        # create the vertices of the mesh
        vertices, _ = self.__model.forward(th_pose_axisang=theta, th_betas=beta, th_trans=None)
        faces = self.__model.th_faces[None, :]
        return vertices, faces
    
    def meshes(self, theta, beta):        
        verts, faces = self.verts_and_faces(theta, beta)
        texture = self.__txmapping(verts, faces)
        mesh = mesh_from(
            vertices=verts, 
            faces=faces, 
            texture=texture)
        
        meshes = mesh.extend(self.__batchsize) if self.__batchsize > 1 else mesh
        return meshes
