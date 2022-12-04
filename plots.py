"""
    This module summarizes our plotting functions  
"""
import matplotlib.pyplot as plt
import seaborn as sns

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
    