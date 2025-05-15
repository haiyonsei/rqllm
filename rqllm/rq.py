
import math
from rqllm.datasets import create_dataset
from rqllm.utils import *
from torch import nn




def soft_quantize(k, CB, tau=0.1, ste=False, mask=None):
    dist2 = ((k.unsqueeze(-2) - CB.unsqueeze(0).unsqueeze(2))**2).sum(-1)
    if mask is not None:
        m = _expand_mask(mask, ndim=dist2.dim())     # pad=True → 마스킹 대상
        dist2 = dist2.masked_fill(m, float('inf'))
    w_soft = (-dist2 / tau).softmax(-1)
    if ste:
        idx = w_soft.argmax(-1, keepdim=True)
        w_hard = torch.zeros_like(w_soft).scatter(-1, idx, 1.0)
        w = (w_hard - w_soft).detach() + w_soft
    else:
        w, idx = w_soft, None
    k_hat = (w.unsqueeze(-1) * CB.unsqueeze(0).unsqueeze(2)).sum(-2)
    return k_hat, w, idx

def init_codebooks_rq(K_, meta, M, L, init='random'):
    B, T, n_head, d_k = meta.B, meta.T, meta.H, meta.D
    with torch.no_grad():
        K_flat = K_.reshape(-1, n_head, d_k)           # (N, H, D)
        codebooks_init = []

        residual = K_flat.clone()                          # 처음엔 실제 K
        for l in range(L):
            CB_stage = torch.empty(n_head, M, d_k)         # (H, M, D)
            for h in range(n_head):
                data_np = residual[:, h, :] #.cpu().numpy()
                # 데이터 수 < M 일 때 대비하여 n_clusters = min(M, len(data))
                n_clust = min(M, data_np.shape[0])
                centers = get_codebook_centers(data_np, M, mode=init)
                #km = KMeans(n_clusters=n_clust, n_init=10, max_iter=100, 
                #            random_state=0) # 64 elements -> 256//L  clusters..  6bit for 64 activations  0.125 bit per activation. 
                #centers = torch.from_numpy(km.fit(data_np).cluster_centers_)
                # 필요한 경우 zero-padding으로 (M,D) 채우기
                if n_clust < M:
                    pad = torch.zeros(M - n_clust, d_k)
                    centers = torch.cat([centers, pad], dim=0)
                CB_stage[h] = centers
            codebooks_init.append(CB_stage)

            k_hat_stage, _, _ = soft_quantize(
                residual.unsqueeze(2),        # → (N, H, 1, D)
                CB_stage,
                tau=1e-9,
                ste=True)
            k_hat_stage = k_hat_stage.squeeze(2)  # → (N, H, D)  원래 차원으로 복구
            residual = residual - k_hat_stage
    return codebooks_init


def train_codebook_core_rq(codebooks_init, Q_, K_, V_, meta, LR=1e-2, EPOCH=1000, TAU=0.1, mask=None):
    B, T, n_head, d_k = meta.B, meta.T, meta.H, meta.D
    codebooks = nn.ParameterList(
        [nn.Parameter(CB.clone()) for CB in codebooks_init]
    )
    optim = torch.optim.Adam(codebooks.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    #def attention(q, k, v):
    #    P = (q @ k.transpose(-2,-1)) / (d_k**0.5)
    #    return P.softmax(-1), torch.matmul(P.softmax(-1), v)

    #def kl_loss(pref, phat, eps=1e-9):
    #    pr = pref.detach()
    #    return (pr * (pr.clamp_min(eps).log() - phat.clamp_min(eps).log())
    #           ).sum(-1).mean()

    # ---------------- 3. 학습 루프 ----------------
    for epoch in range(1, EPOCH+1):
        optim.zero_grad()

        Q = Q_.view(B,T,n_head,d_k).transpose(1,2)
        K = K_.view(B,T,n_head,d_k).transpose(1,2)
        V = V_.view(B,T,n_head,d_k).transpose(1,2)

        with torch.no_grad():
            P_ref,_ = attention(Q, K, V, d_k, mask=mask)


        # --- Residual 양자화 ---
        residual = K
        K_hat_total = 0
        for CB in codebooks:
            K_stage_hat, _, _ = soft_quantize(residual, CB, tau=TAU, ste=True, mask=mask)
            K_hat_total += K_stage_hat
            residual = residual - K_stage_hat          # 다음 단계에 전달

        P_hat,_ = attention(Q, K_hat_total, V, d_k, mask=mask)
        loss = kl_loss(P_ref, P_hat)

        loss.backward()
        optim.step();  scheduler.step()

        if epoch % 10 == 0 or epoch in {1, EPOCH}:
            #att_mse = F.mse_loss(P_hat, P_ref).item()
            #key_mse = F.mse_loss(K_hat_total, K).item()
            att_mse = masked_mse(P_hat, P_ref, mask=mask).item()
            key_mse = masked_mse(K_hat_total, K, mask=mask).item()
            print(f"[epoch {epoch:4d}] KL:{loss.item():.3e}  "
                f"Att-MSE:{att_mse:.3e}  Key-MSE:{key_mse:.3e}  "
                f"LR={optim.param_groups[0]['lr']:.2e}")
    return codebooks


def train_codebook_rq(dataset, M=256, L=1, LR=1e-2, EPOCH=1000, TAU=0.1, init='random'):
    # mask is a (B, T) tensor where 1 indicates invalid tokens
    Q_, K_, V_, meta, mask = create_dataset(dataset)
    B, T, n_head, d_k = meta.B, meta.T, meta.H, meta.D

    #bpa = L*meta.D/math.log2(N)
    #print("BPA : {bpa}")
    #CB_size = M*meta.D
    #print(f"CB size: {CB_size}")
    # padding would be enough for init
    codebooks_init = init_codebooks_rq(K_, meta, M, L, init=init)

    codebooks = train_codebook_core_rq(codebooks_init, Q_, K_, V_, meta, LR, EPOCH, TAU, mask)

    Q = Q_.view(B,T,n_head,d_k).transpose(1,2)
    K = K_.view(B,T,n_head,d_k).transpose(1,2)
    V = V_.view(B,T,n_head,d_k).transpose(1,2)
    with torch.no_grad():
        residual = K; K_hat_total = 0
        for CB in codebooks:
            K_stage_hat, _, _ = soft_quantize(residual, CB, tau=TAU, ste=True, mask=mask)
            K_hat_total += K_stage_hat
            residual = residual - K_stage_hat

        evaluate(Q, K, K_hat_total, V, d_k, mask)
    # codebook 파라미터 등록
