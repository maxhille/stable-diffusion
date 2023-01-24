import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)
