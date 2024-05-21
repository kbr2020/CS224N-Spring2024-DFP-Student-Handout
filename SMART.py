import torch.nn.Functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor




def inf_norm(x):
    return torch.norm(x, p = float('inf'),dim = -1)