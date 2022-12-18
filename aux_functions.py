"""
    This module regroups our auxilary functions
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from torch.nn.functional import cosine_similarity

###########################
# Distance functions
###########################

def cos_dist(u, v):
    return 1 - cosine_similarity(u, v, dim=1, eps=1e-8)

###########################
# Plotters
###########################

def plot_image_t(img_t):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_t[..., :3].detach().cpu().numpy())
    ax.axis("off")
    return fig, ax

def plot_images_t(imgs_t, n, ncols):
    nrows = n // ncols
    nrows += int(n - ncols * nrows > 0)

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    
    for ax, img_t in zip(axs, imgs_t):
        ax.imshow(img_t[..., :3].detach().cpu().numpy())
        ax.axis("off")
    
    return fig, axs

def plot_losses(losses_list, labels_list, colors_list):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    for losses, label, color in zip(losses_list, labels_list, colors_list):
        sns.lineplot(data=losses, x="pass", y="loss", ax=ax, label=label, color=color)
    ax.set_xlabel("pass number")
    ax.set_ylabel("loss")
    ax.legend()
    return fig, ax

def plot_heatmap(array):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(array, linewidth=0.5, ax=ax, cmap="viridis")
    return fig, ax

###########################
# Plotters for Latex
###########################

__RC_PARAMS = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False
    }

def save_for_latex(fig, filename, backend="pgf"):
    if backend == "pgf":
        with mpl.rc_context(__RC_PARAMS):
            fig.savefig(filename+".pgf", backend="pgf", dpi=400)
    else:
        fig.savefig(filename+"png", dpi=400)

###########################
# Info
###########################

def info_str(tensor):
    print("### tensor info:")
    print("shape: " + str(tensor.shape))
    print("device: " + str(tensor.get_device()))
    print("requires grad: " + str(tensor.requires_grad))
    print("### end of tensor info\n")