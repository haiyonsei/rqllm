import math
import torch
from torch import nn
from rqllm.datasets import create_dataset
from rqllm.utils import *
from torch.nn import functional as F
# ---------------- 함수 정의 (PQ 버전) ----------------
def soft_quantize_pq(k, CB, tau=0.1, ste=False):
    B, H, T, D = k.shape
    n_subvec, D_sub = CB.shape[1], CB.shape[-1]
    k = k.view(B, H, T, n_subvec, D_sub)                  # (B,H,T,S,D_sub)
    CB = CB.unsqueeze(0).unsqueeze(2)                     # (1,H,1,S,M,D_sub)

    dist2 = ((k.unsqueeze(-2) - CB)**2).sum(-1)           # (B,H,T,S,M)
    w_soft = (-dist2 / tau).softmax(-1)                   # (B,H,T,S,M)

    if ste:
        idx = w_soft.argmax(-1, keepdim=True)             # (B,H,T,S,1)
        w_hard = torch.zeros_like(w_soft).scatter(-1, idx, 1.0)
        w = (w_hard - w_soft).detach() + w_soft
    else:
        w = w_soft
        idx = None

    CB_exp = CB.squeeze(0).squeeze(1)                     # (H,S,M,D_sub)
    k_hat = (w.unsqueeze(-1) * CB_exp.unsqueeze(0).unsqueeze(2)).sum(-2)
    k_hat = k_hat.view(B, H, T, -1)                       # (B,H,T,D)
    return k_hat, w, idx

# ---------------- 코드북 초기화 (PQ) ----------------
def init_codebooks(Q_, K_, meta, D_sub, n_subvec, M, init='random'):
    B, T, n_head, d_k = meta.B, meta.T, meta.H, meta.D
    with torch.no_grad():
        K_all = K_.reshape(-1, n_head, d_k)     # (B*T, H, D)
        Q_all = Q_.reshape(-1, n_head, d_k)     # (B*T, H, D)
        init_codebooks = torch.empty(n_head, n_subvec, M, D_sub)

        for h in range(n_head):
            for s in range(n_subvec):
                data = K_all[:, h, s*D_sub:(s+1)*D_sub]#.cpu().numpy()
                #km = KMeans(n_clusters=M, n_init=10, random_state=0)
                #ocenters = torch.from_numpy(km.fit(data).cluster_centers_)
                query = Q_all[:, h, s*D_sub:(s+1)*D_sub]#.cpu().numpy()
                centers = get_codebook_centers(data, M, mode=init, query=query)
                init_codebooks[h, s] = centers
    return init_codebooks


# ---------------- 학습 루프 ----------------
def train_codebook_core_pq(init_codebooks, Q_, K_, V_, meta, EPOCH=1000, TAU=0.1, LR=1e-2):
    B, T, n_head, d_k = meta.B, meta.T, meta.H, meta.D
    codebooks = nn.Parameter(init_codebooks.clone())   # (H, n_subvec, M, D_sub)
    optim = torch.optim.Adam([codebooks], lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    # if we train this from corpus directly, these three lines should go to the train loop.
    Q = Q_.view(B, T, n_head, d_k).transpose(1, 2)
    K = K_.view(B, T, n_head, d_k).transpose(1, 2)
    V = V_.view(B, T, n_head, d_k).transpose(1, 2)

    for epoch in range(1, EPOCH+1):
        optim.zero_grad()

        with torch.no_grad():
            P_ref, _ = attention(Q, K, V, d_k)

        K_hat, _, _ = soft_quantize_pq(K, codebooks, tau=TAU, ste=True)
        P_hat, _ = attention(Q, K_hat, V, d_k)
        loss = kl_loss(P_ref, P_hat)

        loss.backward()
        optim.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch in {1, EPOCH}:
            att_mse = F.mse_loss(P_hat, P_ref).item()
            key_mse = F.mse_loss(K_hat, K).item()
            print(f"[epoch {epoch:4d}]  KL:{loss.item():.3e}  "
                f"Att-MSE:{att_mse:.3e}  Key-MSE:{key_mse:.3e} τ={TAU:.3f}  LR={optim.param_groups[0]['lr']:.2e}")
    return codebooks 


def train_codebook_pq(dataset, M=256, n_subvec=1, LR=1e-2, EPOCH=1000, TAU=0.1, init='random'):
    Q_, K_, V_, meta, mask = create_dataset(dataset)
    d_k = meta.D
    D_sub = d_k // n_subvec
    #bpa = n_subvec*d_k/math.log2(M)
    #CB_size = M*D_sub
    #print(f"BPA : {bpa}")
    #print(f"CB size: {CB_size}")
    codebooks = init_codebooks(Q_, K_, meta, D_sub, n_subvec, M, init=init)
    codebooks = train_codebook_core_pq(codebooks, Q_, K_, V_, meta, EPOCH, TAU, LR)
    B, T, n_head, d_k = meta.B, meta.T, meta.H, meta.D


    Q = Q_.view(B, T, n_head, d_k).transpose(1, 2)
    K = K_.view(B, T, n_head, d_k).transpose(1, 2)
    V = V_.view(B, T, n_head, d_k).transpose(1, 2)

    with torch.no_grad():
        K_hat, _, _ = soft_quantize_pq(K, codebooks, tau=TAU, ste=True)
        evaluate(Q, K, K_hat, V, d_k, mask=None)

