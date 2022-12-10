"""
    This module regroups our functions and classes to build the model
"""

def compose(smpl, renderer, clip, **params):
    # the prompt as a hyperparameter
    prompt = params["prompt"]
    
    # smpl mesh creation using the provided smpl model
    def smpl_fn(pose, shape):
        return smpl.mesh(theta=pose, beta=shape)
    
    # mesh rendering using the provided renderer
    def rendering_fn(mesh):
        return renderer.render(mesh)

    # CLIP images embedding
    def clip_imgs_fn(imgs_t):
        return clip.imgs_embs(imgs_t)
    
    # CLIP prompt embedding
    prompt_emb = clip.pmt_emb(prompt)
    
    # the composed model
    def model(pose, shape):
        return clip_imgs_fn(rendering_fn(smpl_fn(pose, shape))), prompt_emb
        
    return model

class SimpledCLIPContext:
    def __init__(self, smpl, renderer, clip):
        self.__smpl = smpl
        self.__renderer = renderer
        self.__clip = clip
        
    def create(self,  prompt):
        return compose(self.__smpl, self.__renderer, self.__clip, prompt=prompt)