"""
    This module regroups our functions and classes to build the model
"""

def compose(smpl, renderer, clip, **params):
    """
    Composes an instance of SMPL, the differentiable renderer and CLIP
    with a hyperparameter prompt, into a single instance of SimpledCLIP 
    (in the form of a Callable).

    Args:
        smpl: an instance of an SMPL model.
        renderer: an instance of a differentiable renderer.
        clip: an instance of a CLIP model.
        **params:
            prompt: the prompt containing a pose description to be recovered.

    Returns:
        A SimpledCLIP model, in the form of a Callable
    """
    # the prompt as a hyperparameter
    prompt = params["prompt"]
    
    # smpl mesh creation using the provided smpl model
    def smpl_fn(pose, shape):
        return smpl.meshes(theta=pose, beta=shape)
    
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
    """
    Represents a context for creating SimpledCLIP instances using the same models.
    """
    def __init__(self, smpl, renderer, clip):
        self.__smpl = smpl
        self.__renderer = renderer
        self.__clip = clip
        
    def get_model(self,  prompt):
        """
        Creates a SimpledCLIP model for a given prompt using the underlying models.
        
        Args:
            prompt: the prompt containing a pose description to be recovered.
            
        Returns:
            A SimpledCLIP model, in the form of a Callable
        """
        return compose(self.__smpl, self.__renderer, self.__clip, prompt=prompt)