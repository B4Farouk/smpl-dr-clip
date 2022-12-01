"""
    This module regroups our functions and classes to build the
"""
import torch

def build_model(smpl, renderer, clip_model):
    def smpl_fn(pose, shape):
        mesh = smpl.mesh(theta=pose, beta=shape)
        return mesh
    
    def renderer_fn(mesh):
        return renderer.render(mesh)
    
    def clip_fn(image, prompt):
        # formatting the image to fit the input
        image = image.squeeze()
        image = torch.permute(image, (2, 0, 1))
        image = image[:3,:,:]
        
        # For one text and one image
        similarity = clip_model.get_cosine_similarity(image, prompt)
        return similarity[0][0]
    
    def model(pose, shape, prompt):
        return clip_fn(renderer_fn(smpl_fn(pose, shape)), prompt)
        
    return model