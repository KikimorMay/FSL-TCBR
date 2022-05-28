import math

import torch
from torch import nn as nn
from torch.nn import Parameter, functional as F

def get_cos_sin( x, y):
    cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
    sin_val = (1 - cos_val * cos_val).sqrt()
    return cos_val, sin_val

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)

class CosineSimilarity(nn.Module):
    def __init__(self, in_features, out_features, scale_factor=30.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.Tensor(out_features, in_features).float())
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine * self.scale_factor

class CosineSimilarity_miuns(nn.Module):
    def  __init__(self, in_features, out_features, scale_factor=30.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.Tensor(out_features, in_features).float())
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, feature_base):
        feature = F.normalize(feature)
        cos_val, sin_val = get_cos_sin(feature, feature_base)

        feature = feature - cos_val*feature_base
        cosine = F.linear(feature, F.normalize(self.weight))

        return cosine * self.scale_factor

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    :return： (theta) - m
    """

    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.40):
        super(AddMarginProduct, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))

        # when test, no label, just return
        if label is None:
            return cosine * self.scale_factor

        phi = cosine - self.margin
        output = torch.where(
            one_hot(label, cosine.shape[1]).byte(), phi, cosine)
        output *= self.scale_factor

        return output

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()
        return acc

class SoftmaxMargin(nn.Module):
    r"""Implement of softmax with margin:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    """

    def __init__(self, in_features, out_features, scale_factor=5.0, margin=0.40):
        super(SoftmaxMargin, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        z = F.linear(feature, self.weight)
        z -= z.min(dim=1, keepdim=True)[0]
        # when test, no label, just return
        if label is None:
            return z * self.scale_factor

        phi = z - self.margin
        output = torch.where(
            one_hot(label, z.shape[1]).byte(), phi, z)
        output *= self.scale_factor

        return output
