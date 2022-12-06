import torch
import PIL
import numpy as np
import trimesh
from pytorch3d.renderer import (
    TexturesAtlas,
    TexturesUV,
    TexturesVertex
)

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
    
    def from_image(self,colored_reference_SMPL,verts,faces):
        
        random_SMPL = trimesh.Trimesh(verts[0].cpu(), faces.cpu(), process=False)
        random_SMPL.visual.vertex_colors = colored_reference_SMPL.visual.vertex_colors
        texture = torch.from_numpy(
            colored_reference_SMPL.visual.vertex_colors[:,:3] # Remove transparency
          ).unsqueeze(0) / 255 # Add a fake batch size and normalize to [0,1]
        textures = TexturesVertex(verts_features=texture)
        textures.to(self.__device)
        return texture
