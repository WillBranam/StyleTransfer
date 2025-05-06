import torch
import torch.nn.functional as F

def gram_matrix(y):
    (b, c, h, w) = y.size()
    features = y.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

def compute_style_loss(features_y, features_s):
    return sum(F.mse_loss(gram_matrix(y), gram_matrix(s)) for y, s in zip(features_y, features_s))

def compute_content_loss(features_y, features_c):
    return F.mse_loss(features_y[-1], features_c[-1])
