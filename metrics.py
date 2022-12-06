"""
    This module regroups our metrics
"""
import torch.nn as nn 

###########################
# Similarities
###########################

def sim(u, v):
    return u @ v

__COSINE_SIMILARITY = nn.CosineSimilarity(dim=1, eps=1e-8)
def cos_sim(u, v):
    return __COSINE_SIMILARITY(u, v)

###########################
# Distances
###########################

def cos_dist(u, v):
    return 1 - cos_sim(u, v)

__MSE_LOSS = nn.MSELoss()
def mse(u, v):
    return __MSE_LOSS(u, v)