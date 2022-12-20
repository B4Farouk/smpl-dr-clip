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
    """
    cosine distance between two tensors.

    Args:
        u (torch.Tensor): the first tensor
        v (torch.Tensor): the second tensor

    Returns:
        (torch.Tensor): a scalar tensor in the interval [0, 1]
    """
    return 1 - cosine_similarity(u, v, dim=1, eps=1e-8)

###########################
# Plotters
###########################

def plot_image_t(img_t):
    """
    plots an image

    Args:
        img_t (torch.Tensor): an image tensor of shape (W, H, 3) to be plotted

    Returns:
        a figure and its axis
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_t[..., :3].detach().cpu().numpy())
    ax.axis("off")
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

def save_for_latex(fig, filename, bbox_inches="tight", backend="pgf"):
    if backend == "pgf":
        with mpl.rc_context(__RC_PARAMS):
            fig.savefig(filename+".pgf", backend="pgf", dpi=400, bbox_inches=bbox_inches)
    else:
        fig.savefig(filename+"png", dpi=400, bbox_inches=bbox_inches)
