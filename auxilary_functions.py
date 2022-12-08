"""
    This module regroups our auxilary functions
"""
from torch.nn.functional import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

###########################
# Similarities
###########################

def cos_sim(u, v):
    return cosine_similarity(u, v, dim=1, eps=1e-8)

###########################
# Distances
###########################

def cos_dist(u, v):
    return 1 - cos_sim(u, v)

###########################
# Plotters
###########################

def plot_image_t(img_t):
    _, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.imshow(img_t[0, ..., :3].detach().cpu().numpy())
    ax.axis("off")
    return ax

def plot_losses(losses):
    _, ax = plt.subplots(1, 1, figsize=(10,10))
    sns.lineplot(data=losses, x="pass", y="loss", ax=ax)
    ax.set_title("Distance Between Image and Prompt Embeddings")
    ax.set_xlabel("pass number")
    ax.set_ylabel("distance")
    return ax