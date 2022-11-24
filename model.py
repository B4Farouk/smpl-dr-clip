"""
    This module regroups our functions and classes to build the
"""

def build_model(smpl, renderer, clip):
    def smpl_fn(pose, shape):
        vertices, _ = smpl(pose, shape)
        return vertices
    
    def renderer_fn(mesh):
        return renderer.render(mesh)
    
    def clip_fn(image, prompt):
        # TODO
        pass
    
    def model(pose, shape, prompt):
        return clip_fn(renderer_fn(smpl_fn(pose, shape)), prompt)
        
    return model