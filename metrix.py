"""
    This module regroups our metrics
"""
from torch.nn.functional import cosine_similarity

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