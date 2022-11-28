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

class TexturesFactory:
    def __init__(self, device):
        self.__device = device
    
    def from_facecolor(self, nfaces, facecolor):
        # check input
        assert nfaces > 0 and len(facecolor) == 3
        
        # create an atlas
        facecolor = torch.Tensor(facecolor)
        white = torch.ones((nfaces, 3))
        facecolors = facecolor * white
        atlas = facecolors[None, :, None, None, :] # (#meshs=1, #faces, ?=1, 1=1, RGB_colors)
                
        # create a texture
        textures = TexturesAtlas(atlas=atlas)
        textures.to(self.__device)
        return textures
        
    def from_facecolors(self, facecolors):
        # check input
        assert facecolors is not None and facecolors.shape[1] == 3
        
        # create an atlas
        atlas = facecolors[None, :, None, None, :] # (#meshs=1, #faces, ?=1, 1=1, RGB_colors)
        
        # create a texture
        textures = TexturesAtlas(atlas=atlas)
        textures.to(self.__device)
        return textures

class SMPL:
    __DEFAULT_MODEL = SMPL_Layer(model_root='/content/project')
    
    def __init__(self, model, device):
        device = device if device is not None else torch.device("cpu")
        self.__model = model if model is not None else SMPL.__DEFAULT_MODEL
        self.__model.to(device)
             
    # theta is the pose parameter of shape (1,72) 
    # beta is the shape parameter of shape (1,10)
    def verts_and_faces(self, beta, theta):
        # create the vertices of the mesh
        vertices, _ = self.__model.forward(th_pose_axisang=theta, th_betas=beta, th_trans=None)
        faces = self.__model.th_faces[None, :]
        return vertices, faces
    
    def mesh(self, beta, theta, txmapping):
        verts, faces = self.verts_and_faces(beta, theta)
        texture = txmapping(faces) # a function that creates a texture from faces
        mesh = SMPL.mesh_from(
            vertices=verts, 
            faces=faces, 
            texture=texture)
        return mesh