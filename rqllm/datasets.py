import torch
from torch import nn
from types import SimpleNamespace

def create_dataset(name):
    if name == 'random':
        
        B, T, d_model, n_head = 64, 128, 256, 4
        #B, T, d_model, n_head = 8, 128, 256, 4
        d_k = d_model // n_head         # 64
        meta = SimpleNamespace(B=B, T=T, H=n_head, D=d_k)
        with torch.no_grad():
            x = torch.randn(B, T, d_model)
            W_Q, W_K, W_V = nn.Linear(d_model, d_model, bias=False), nn.Linear(d_model, d_model, bias=False), nn.Linear(d_model, d_model, bias=False)
            return W_Q(x), W_K(x), W_V(x), meta, None
