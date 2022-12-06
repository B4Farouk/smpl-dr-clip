import torch
import PIL
import numpy as np

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
    
    def from_image(self,texture_map):
        texture_map = PIL.Image.open(texture_map)
        texture_map = np.array(texture_map,dtype=np.float) / 255.0
        texture_map = torch.tensor(texture_map,dtype=torch.float).permute(2,0,1).unsqueeze(0)
        textures = TexturesAtlas(atlas=texture_map)
        textures.to(self.__device)
        return texture
