# Models and Parameters

This file documents our model version choice and the best hyperparameters we could find, which we used to produce the results in our paper.

## SMPL

model: https://github.com/gulvarol/smplpytorch

mesh_gender: "female"

## Texture

human texture: https://github.com/B4Farouk/smpl-dr-clip/blob/main/SMPL_female_with_colors.obj

grey texture: set the color of all the mesh faces to grey corresponding to rgba(0.5, 0.5, 0.5, 1)

## Differentiable Renderer

implementation: pytorch3d (v 0.7.1)

### Camera(s)

#### Single Camera Rendering

camera type: field of view

specifications: radius=2.25, azimuth=0°, elevation=5°, field_of_view=60°

#### Multi-Camera Rendering

camera types: field of view

specifications:
- camera 1: radius=2.25, azimuth=0°, elevation=5°, field_of_view=60°
- camera 2: radius=2.25, azimuth=45°, elevation=5°, field_of_view=60°
- camera 3: radius=2.25, azimuth=135°, elevation=5°, field_of_view=60°
- camera 4: radius=2.25, azimuth=-45°, elevation=5°, field_of_view=60°
- camera 5: radius=2.25, azimuth=-135°, elevation=5°, field_of_view=60°

### Rasterizer

settings: image_size=(244, 244), blur_radius=0.0, faces_per_pixel=1, bin_size = None,
max_faces_per_bin = None, perspective_correct = None, clip_barycentric_coords = None, z_clip_value = None,
cull_backfaces = False, cull_to_frustum = False

### Shader

type: hard flat shader

settings: Lights=None, Materials=None, Blending=None

## CLIP

model: "ViT-B/32"

channels_normalization: z_score{mean=(0.48145466, 0.4578275, 0.40821073), variance=(0.26862954, 0.26130258, 0.27577711)}

## Optimization

### Optimizer
optimizer: Adam, lr= $10^{-3}$, betas= $(0.9, 0.999)$

### LR Scheduler
We tried to use a ReduceOnPlateau learning rate scheduler without success. The default parameters are specified within the OptimConfig class of optimizaiton.py.

## Loss
### Inner Product Loss
$\lambda = 10^{-3}$
