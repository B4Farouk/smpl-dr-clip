"""
    This module abstracts away the implementation details of the differentiable renderer
    used in this project.
"""

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

class Renderer:    
    def __init__(self, rasterizer, shader):
        self.__renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        
    def apply(self, mesh):
        return self.__renderer(mesh)