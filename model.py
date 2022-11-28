"""
    This module regroups our functions and classes to build the
"""

def build_model(smpl, renderer, clip_model):
    def smpl_fn(pose, shape):
        vertices, _ = smpl(pose, shape)
        return vertices
    
    def renderer_fn(mesh):
        return renderer.render(mesh)
    
    def clip_fn(image, prompt):
        # For one text and one image
        similarity = clip_model.get_cosine_similarity(list(image), list(prompt))
        return similarity[0][0]
    
    def model(pose, shape, prompt):
        return clip_fn(renderer_fn(smpl_fn(pose, shape)), prompt)
        
    return model