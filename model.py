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
    def renderer_fn(mesh):
        return renderer.render(mesh)

    # CLIP image embedding
    def clip_img_fn(img_t):
        return clip.image_embedding(img_t)
    
    # CLIP prompt embedding
    def clip_prompt_fn(prompt):
        return clip.prompt_embedding(prompt)
    
    # compute the prompt embedding once only
    prompt_emb = clip_prompt_fn(prompt)
    
    # the composed model
    def model(pose, shape):
        return clip_img_fn(renderer_fn(smpl_fn(pose, shape))), prompt_emb
        
    return model