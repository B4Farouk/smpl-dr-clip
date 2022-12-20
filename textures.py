"""
    This module contains our functions and classes to generate textures for meshes.
"""

import torch
import trimesh
from pytorch3d.renderer import (
    TexturesAtlas,
    TexturesVertex
)

class TexturesFactory:
    """
    A factory class for mesh textures.
    """
    def __init__(self, device):
        self.__device = device
    
    def from_facecolor(self, nfaces, facecolor):
        """
        Creates a mono-color texture.

        Args:
            nfaces: the number of faces in the mesh.
            facecolor: a color in the form of an interable of 3 RGB values.

        Returns:
            A texture
        """
        # check input
        assert nfaces > 0 and len(facecolor) == 3
        
        # create an atlas
        facecolor = torch.Tensor(facecolor)
        white = torch.ones((nfaces, 3))
        facecolors = facecolor * white
        atlas = facecolors[None, :, None, None, :]
                
        # create a texture
        textures = TexturesAtlas(atlas=atlas)
        textures.to(self.__device)
        return textures
        
    def from_facecolors(self, facecolors):
        """
        Creates a texture from a collection of face colors.

        Args:
            facecolors: a collection of colors, where each color is in the form of an interable of 3 RGB values. 
                        The collection must have as many entries as the number of faces in the mesh. 

        Returns:
            A texture
        """
        # check input
        assert facecolors is not None and facecolors.shape[1] == 3
        
        # create an atlas
        atlas = facecolors[None, :, None, None, :]
        
        # create a texture
        textures = TexturesAtlas(atlas=atlas)
        textures.to(self.__device)
        return textures
    
    def from_image(self,colored_reference_SMPL,verts,faces):
        """
        Creates a texture from a trimesh,vertices and faces.
        Args:
            colored_reference_SMPL: Trimesh object from trimesh
            verts: lists of vertices
            faces: lists of faces
        Returns:
            A texture
        """
        random_SMPL = trimesh.Trimesh(verts[0].detach().to(torch.device("cpu")), faces.detach().to(torch.device("cpu")), process=False)
        random_SMPL.visual.vertex_colors = colored_reference_SMPL.visual.vertex_colors
        texture = torch.from_numpy(
            colored_reference_SMPL.visual.vertex_colors[:,:3] # Remove transparency
          ).unsqueeze(0) / 255 # Add a fake batch size and normalize to [0,1]
        texture = TexturesVertex(verts_features=texture)
        texture.to(self.__device)
        return texture
