import torch
from torch._C import dtype
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from typing import Tuple

class VisionPonder(nn.Module):
    def __init__(self, model: nn.Module, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        
        self.model = model
        
        self.lambda_layer = nn.Linear(self.model.mlp_dim, 1)
        self.lambda_prob = nn.Sigmoid()
        self.n_classes = self.model.n_classes
        self.n_hidden = self.model.dim
        self.mlp_head = self.model.mlp_head

    def mlp_layer(self, x:torch.Tensor):
        x = x[:, 0]
        x = self.mlp_head(x)
        return x

    def forward(self, x: torch.Tensor, is_halt = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        # find initial value of h
        h = self.model(x)
        #h = self.enc(h)
        #print("h2: ",h.shape)
        p = []
        y = []
        un_halted_prob = h.new_ones((batch_size,))
        halted = h.new_zeros((batch_size,))
        p_m = h.new_zeros((batch_size,))
        y_m = h.new_zeros((batch_size,self.n_classes))
        
        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = h.new_ones(h.shape[0])
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(h))[:, 0]
            lambda_n = lambda_n.squeeze()

            y_n = self.mlp_layer(h)
            p_n = un_halted_prob * lambda_n
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            halt = torch.bernoulli(lambda_n) * (1 - halted)
            
            p.append(p_n)
            y.append(y_n)
            p_m = p_m * (1 - halt) + p_n * halt
            
            y_m = y_m * (1 - halt.unsqueeze(1)) + y_n * halt.unsqueeze(1)

            halted = halted + halt
            h = self.model.transformer(h)
            """
            print("p_n: ", p_n.shape)
            print("p_m: ", p_m.shape)
            print("halt: ", halt.shape)


            print("y_n: ", y_n.shape)
            print("y_m: ", y_m.shape)
            print("halt: ", halt.unsqueeze(1).shape)
            """
            #print("halted sum", self.is_halt and int(halted.sum().item()) == batch_size)
            if is_halt and int(halted.sum().item()) == batch_size:
                break
        # h = self.to_cls_token([:, 0])
        # self.mlp_head()
        return torch.stack(p), torch.stack(y), p_m, y_m

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p: torch.Tensor, y_hat: torch.Tensor, y:torch.Tensor):
        total_loss = p.new_tensor(0.)
        for n in range(p.shape[0]):
            #loss = (p[n] * self.loss_func(y_hat[n], y)).mean()
            loss = (self.loss_func(y_hat[n], y)).mean()
            total_loss = total_loss + loss
        return total_loss

class RegularizationLoss(nn.Module):
    def __init__(self, lambda_p: float, max_steps: int = 1_000):
        super().__init__()
        p_g = torch.zeros((max_steps,))
        not_halted = 1.

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        #self.kl_div = torch.distributions.kl_divergence
    def forward(self, p: torch.Tensor):
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        # kl divergence
        kl_v = p * (torch.div(p,p_g)).log()
        mean = kl_v.mean()
        # mean = kl_v[torch.isnan(kl_v) !=True].mean()
        print(p)
        return mean
        # return self.kl_div(p.log(), p_g)
        #return self.kl_div(p, p_g).mean()
        #return torch.distributions.kl_divergence(p, p_g).mean()