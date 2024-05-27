


import torch
from torch import nn
import torch.nn.functional as F
from itertools import count 




def inf_norm(x):
    return torch.norm(x, p = float('inf'),dim = -1)

def kl_loss(input, target, reduction='batchmean'):
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )

def sym_kl_loss(input, target, reduction='sum', alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )



class SMART_loss(nn.Module):
    def __init__(self, eval_func, loss_func, norm_fn =  inf_norm,   num_steps: int = 1, step_size: float = 1e-3, epsilon: float = 1e-6,noise_var: float = 1e-5):
    
        super().__init__()
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        self.eval_func = eval_func
        self.loss_func = loss_func
        self.norm_fn = norm_fn


    def forward(self, embedd , state):
        noise = torch.randn_like(embedd, requires_grad=True)*self.noise_var

        for i in count():
            embedd_noise = embedd + noise
            state_no = self.eval_func(embedd_noise)

            if i == self.num_steps: 
                return self.loss_func(state_no, state) 

            loss = self.loss_func(state_no, state.detach())

            noise_gr= torch.autograd.grad(loss, noise)

            noise_gr = noise_gr[0]

            step = noise + self.step_size * noise_gr
            print(step.shape)
            step_norm = self.norm_fn(step)
            print(step_norm.shape)
            noise = step / (step_norm.view(-1,1) + self.epsilon)

            noise = noise.detach().requires_grad_() 
        
    