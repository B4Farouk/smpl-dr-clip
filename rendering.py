"""
    This module abstracts away the implementation details of the differentiable renderer
    used in this project.
"""

import numpy as np

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

class CamerasFactory:
    def __init__(self, device):
        self.__device = device
        
    def fov_persp_scs(self, coords, fov, frustrum_depth, degrees: bool =True):
        """
        Creates one or multiple cameras defined by their spherical coordinates, field of view
        and viewing frustrum. Each of these cameras' field of view is centered at the origin.

        Args:
            coords (Tuple[float, float, float]): spherical coordinates.
            fov (float): field of view.
            frustrum_depth (Tuple[float, float]): maximum and minimum depth (Z coordinate) of the frustrum.
            degrees (bool, optional): Switch between degrees and radians. Defaults to True.

        Returns:
            (FoVPerspectiveCameras): one or multiple FoV cameras.
        """
        # camera coordinates in Spherical Coordinate System
        # radius > 0
        # elevation in [-90°, 90°]
        # azimuthal angle in [-180°, 180°] or [0°, 360°]
        radius, elev, azim = coords
        
        # computes coordinate transforms from WCS to VCS
        rotation, translation = look_at_view_transform(dist=radius, elev=elev, azim=azim, degrees=degrees)
        
        # decoupling the frustrum parameters
        min_depth, max_depth = frustrum_depth
        assert np.all(min_depth > 0) & np.all(max_depth > 0)
        
        # creating the Field Of View Perspective Camera(s)
        cameras = FoVPerspectiveCameras(
            znear=min_depth,
            zfar=max_depth,
            fov=fov,
            R=rotation, 
            T=translation,
            device=self.__device)
        
        return cameras

class RasterizerFactory:
    def __init__(self, device):
        self.__device = device
        
    

class Renderer:    
    def __init__(self, rasterizer, shader):
        self.__renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        
    def apply(self, mesh):
        return self.__renderer(mesh)