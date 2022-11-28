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

class TexturesFactory:
    def __init__(self, device):
        self.__device = device
    
    def from_nfaces(self, n, color):
        # check input
        assert n > 0 and len(color) == 3
        
        # create an atlas
        face_colors = color * torch.ones((n, 3), device=self.__device)
        atlas = face_colors[None, :, None, None, :] # (#meshs=1, #faces, ?=1, 1=1, RGB_colors)
                
        # create a texture
        textures = TexturesAtlas(atlas=atlas)
        textures.to(self.__device)
        return textures
        
    def from_face_colors(self, face_colors):
        # check input
        assert face_colors is not None and face_colors.shape[1] == 3
        
        # create an atlas
        atlas = face_colors[None, :, None, None, :] # (#meshs=1, #faces, ?=1, 1=1, RGB_colors)
        
        # create a texture
        textures = TexturesAtlas(atlas=atlas)
        textures.to(self.__device)
        return textures

class SMPL:
    __DEFAULT_MODEL = SMPL_Layer(model_root='/content/project')
    
    def __init__(self, device, model, textures):
        self.__device = device if device is not None else torch.device("cpu")
        self.__model = model if model is not None else SMPL.__DEFAULT_MODEL
        
    def to(self, device):
        assert device is not None
        self.__model.to(device)
        
    # theta is the pose parameter of shape (1,72) 
    # beta is the shape parameter of shape (1,10)
    def verts_and_faces(self, betas, thetas):
        # move the model to the device
        self.__model.to(self.__device)
        # create the vertices of the mesh
        vertices, _ = self.__model.forward(th_pose_axisang=thetas, th_betas=betas, th_trans=None)
        faces = self.__model.th_faces[None, :]
        return vertices, faces
    
    def meshes_from(self, vertices, faces, textures):
        # create the mesh
        mesh = Meshes(
            verts=vertices, 
            faces=faces,
            textures=textures)
        return mesh
        