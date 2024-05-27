


import torch
from torch import nn
from torch import Tensor
import torch.nn.Functional as F





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
    def __init__(self, eval_func, loss_func,    num_steps: int = 1, step_size: float = 1e-3, epsilon: float = 1e-6,noise_var: float = 1e-5):
    
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
            state_no = self.feval_fn(embedd_noise)

            loss = self.loss_func(state_no, state.detach())

            noise_gr, _ = torch.autograd.grad(loss, noise)

            step = noise + self.step_size * noise_gr

            step_norm = self.norm_fn(step)

            noise = step / (step_norm + self.epsilon)

            noise = noise.detach().requires_grad_() 
        
    