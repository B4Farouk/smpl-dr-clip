"""
    This module abstracts away the implementation details of the differentiable renderer
    used in this project.
"""
import numpy as np

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    
    RasterizationSettings, 
    MeshRasterizer,
    
    MeshRenderer, 
    
    HardFlatShader,
)
    
class CamerasFactory:
    """
    Factory class for Cameras
    """
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
        radius, azim, elev = coords
        
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

class Renderer:
    """
    Differentiable Renderer class
    """
    __DEFAULT_RASTERIZER_SETTINGS = RasterizationSettings(
    image_size = (244, 244), # height then width
    
    blur_radius = 0.0, # no blurring effect wanted
    
    faces_per_pixel = 1, # how many faces can share a pixel at the same time if there is an overlapping
    
    # binning
    bin_size = None,
    max_faces_per_bin = None,
    
    # perspective
    perspective_correct = None, # perspective correction for barycentric coordinate computation, here does so if camera uses perspective
    
    # clipping
    clip_barycentric_coords = None, # clips barycentric coords if outside the face
    z_clip_value = None, # clips depth coords to avoid infinite projections
    
    # culling
    cull_backfaces = False,
    cull_to_frustum = False
    )
    
    def __init__(self, 
                 device, 
                 cameras, 
                 rasterization_settings = None,
                 shader = None):
        # rasterizer
        if rasterization_settings is None:
            self.__rasterizer = MeshRasterizer(cameras, Renderer.__DEFAULT_RASTERIZER_SETTINGS)
        else:
            self.__rasterizer = MeshRasterizer(cameras, rasterization_settings)
        # shader
        self.__shader = shader if shader is not None \
            else HardFlatShader(
                device = device, 
                cameras = cameras,
                lights = None,
                materials = None,
                blend_params = None
            )
        # renderer
        self.__renderer = MeshRenderer(rasterizer=self.__rasterizer, shader=self.__shader)
        self.__renderer.to(device)
    
    def render(self, meshes):
        """
        Renders a mesh

        Args:
            meshes: meshes to be rendered.

        Returns:
            The resulting images.
        """
        return self.__renderer(meshes)