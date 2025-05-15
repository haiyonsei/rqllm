

import torch, torch.nn.functional as F
from torch import nn
from sklearn.cluster import KMeans
from rqllm.utils import kl_loss





def attention_for_query_weighted_kmeans(q, k, v, d_k):
    att = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)      # (B,H,T,T)
    P = att.softmax(-1)
    out = P @ v
    return P, out

#def kl_loss(pref, phat, eps=-1e-9):
#    pref_ = pref.detach()
#    return (pref_ * (pref_.clamp_min(eps).log() - phat.clamp_min(eps).log())).sum(-3).mean()

def _normalize_to_3d(mat, H):
    """
    mat 이 (D,D) → (1,D,D) → (H,D,D) 로 복제
        혹은 이미 (H,D,D) 면 그대로 반환
    """
    if mat.dim() == 2:
        mat = mat.unsqueeze(0)        # (1,D,D)
    if mat.size(0) == 1 and H > 1:
        mat = mat.expand(H, -1, -1)   # (H,D,D)
    return mat                        # (H,D,D)

def encode_K(codebooks, whiten, k):                      # k: (B,H,T,D)
    B, H, T, D = k.shape
    k = k.permute(0, 2, 1, 3)         # (B,T,H,D)

    W = _normalize_to_3d(whiten, H)   # (H,D,D) 확정
    # ---------- 디버깅 출력 ----------
    if False:  # True 로 바꾸면 shape log
        print("encode_K shapes →",
              "W:", W.shape, "k:", k.shape)

    k_tilde = torch.einsum('hij,bthj->bthi', W, k)    # (B,T,H,D)

    diff = k_tilde.unsqueeze(-2) \
         - codebooks.unsqueeze(0).unsqueeze(0)        # (B,T,H,1,D) vs (1,1,H,M,D)
    idx  = diff.pow(2).sum(-1).argmin(-1)             # (B,T,H)
    return idx.permute(0, 2, 1)                       # (B,H,T)

def decode_K(codebooks, Sigma_Q, idx):                       # idx: (B,H,T)
    B, H, T = idx.shape
    idx_bt_h = idx.permute(0, 2, 1)      # (B,T,H)

    # (1) codebooks  → (1,1,H,M,D)  로 차원 맞추기
    C_all = codebooks.unsqueeze(0).unsqueeze(0)        # (1,1,H,M,D)

    # (2) idx        → (B,T,H,1,1)  로 expand
    idx_exp = idx_bt_h.unsqueeze(-1).unsqueeze(-1)     # (B,T,H,1,1)
    idx_exp = idx_exp.expand(-1, -1, -1, 1, codebooks.size(-1))

    # (3) M(=3-rd) 축에서 gather → (B,T,H,1,D)  → squeeze
    C = torch.gather(C_all.expand(B, T, -1, -1, -1),
                     dim=3, index=idx_exp).squeeze(3)  # (B,T,H,D)

    # (4) un-whiten  (H,D,D) or (1,D,D) 모두 대응
    S = _normalize_to_3d(Sigma_Q, H)                   # (H,D,D)
    k_hat = torch.einsum('hij,bthj->bthi',
                         torch.linalg.cholesky(S), C)  # (B,T,H,D)

    return k_hat.permute(0, 2, 1, 3)                   # (B,H,T,D)


def query_weighted_kmeans(Q_, K_, V_, meta, M=256):

    B, T, n_head, d_k = meta.B, meta.T, meta.H, meta.D
    # 1. 코드북 학습 (Query-가중 K-means)
    # ------------------------------
    with torch.no_grad():
        Q = Q_.reshape(-1, n_head, d_k)          # (B*T, H, d_k)
        K = K_.reshape(-1, n_head, d_k)

        # Σ_Q = E[QᵀQ]/d_k  →  whitening 변환으로 가중 치우침 반영
        Sigma_Q = (Q.transpose(-1, -2) @ Q).mean(dim=0) / d_k     # (H, d_k, d_k)
        whiten = torch.linalg.inv(torch.linalg.cholesky(Sigma_Q)) # (H, d_k, d_k)
        K_tilde = (whiten @ K.unsqueeze(-1)).squeeze(-1)           # (N,H,d_k)

        codebooks = []
        assignments = []

        for h in range(n_head):
            kmeans = KMeans(n_clusters=M, n_init=10, max_iter=100)
            kmeans.fit(K_tilde[:, h, :].cpu().numpy())
            C_h = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32,
                            device=Q_.device)                 # (M,d_k)
            codebooks.append(C_h)
            assignments.append(torch.tensor(kmeans.labels_, device=Q_.device))

        codebooks = torch.stack(codebooks)                      # (H,M,d_k)

        Q_full = Q_.view(B, T, n_head, d_k).transpose(1, 2)  # (B,H,T,d_k)
        K_full = K_.view(B, T, n_head, d_k).transpose(1, 2)
        V_full = V_.view(B, T, n_head, d_k).transpose(1, 2)
        P_ref, _ = attention_for_query_weighted_kmeans(Q_full, K_full, V_full, d_k)             # (B,H,T,T)

        # (b) 즉시 KV 양자화 & score 계산 (Query는 여전히 FP)
        K_idx = encode_K(codebooks, whiten, K_full)                                 # (B,H,T)
        K_hat = decode_K(codebooks, Sigma_Q, K_idx)                                  # (B,H,T,d_k)
        P_hat, _ = attention_for_query_weighted_kmeans(Q_full, K_hat, V_full, d_k)

        # -------- MSE --------
        mse = F.mse_loss(P_hat, P_ref).item()
        # row-wise KL : sum_j P_ref * log(P_ref/P_hat)
        kl = kl_loss(P_ref, P_hat)


    print(f"MSE(attention) = {mse:.4e}")
    print(f"KL  (attention) = {kl : .4e}")