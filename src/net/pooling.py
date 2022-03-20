import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


def gem_pool(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class L2N(nn.Module):
    def __init__(self, eps=1e-6, **kwargs):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + f'(eps={self.eps})'


class GEM(nn.Module):
    def __init__(self, p=3, eps=1e-6, **kwargs):
        super(GEM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.l2 = L2N()

    def forward(self, x):
        global_descriptors = gem_pool(self.l2(x), p=self.p, eps=self.eps)
        return global_descriptors

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})'
