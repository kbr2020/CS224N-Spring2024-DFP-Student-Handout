import torch.nn.Functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor




def inf_norm(x):
    return torch.norm(x, p = float('inf'),dim = -1)



class SMART_loss(nn.Module):
    def __init(self, eval_func, loss_func    num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5):

        super().__init__()
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        self.eval_dunc = eval_func
        self.loss_func = loss_func


    def fowrward(self, embedd , state):
        noise = nn.randn_like(embedd, require_grad = True)*self.noise_var

        for i in range(self.num_steps):
            embedd_noise = embedd + noise
            state_perturbated = self.feval_fn(embedd_noise)

            loss = self.loss_func(state_perturbated, state.detach())





    def forward(self):
    