

import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F


#def kl_loss(pref, phat, eps=-1e-9):
#    pref_ = pref.detach()
#    return (pref_ * (pref_.clamp_min(eps).log() - phat.clamp_min(eps).log())).sum(-4).mean()


def _expand_mask(mask: torch.Tensor, *, ndim: int, new_axis: int = -1):
    """
    mask: (...,)
    반환: (..., 1) - ndim 차원까지 1-expansion
    """
    for _ in range(ndim - mask.dim()):
        mask = mask.unsqueeze(new_axis)
    return mask


def attention(q, k, v, d_k, mask=None):
    P = (q @ k.transpose(-2,-1)) / (d_k**0.5)
    if mask is not None:
        m = _expand_mask(mask, ndim=score.dim(), new_axis=-2) 
        P = P.masked_fill(m, float('-inf'))
    return P.softmax(-1), torch.matmul(P.softmax(-1), v)


def masked_mse(a, b, mask, eps=1e-9):
    if mask is None:
        return F.mse_loss(a, b)
    diff2 = (a - b) ** 2
    m     = _expand_mask(mask, ndim=diff2.dim())
    return (diff2 * ~m).sum() / (m.sum() + eps)


def evaluate(Q, K, K_hat, V, d_k, mask):
    # ---------------- 최종 평가 ----------------
    print("\n===== [FINAL EVALUATION] =====")
    with torch.no_grad():
        P_ref, _ = attention(Q, K, V, d_k, mask=mask)
        P_hat, _ = attention(Q, K_hat, V, d_k, mask=mask)

        final_kl   = kl_loss(P_ref, P_hat).item()
        final_amse = masked_mse(P_hat, P_ref, mask=mask).item()
        final_kmse = masked_mse(K_hat, K, mask=mask).item()

    print(f"Final KL      : {final_kl:.3e}")
    print(f"Final Att-MSE : {final_amse:.3e}")
    print(f"Final Key-MSE : {final_kmse:.3e}")


def get_codebook_centers(data, M, mode='kmeans', query=None):
    """
    data : (N, D)            - 특정 head의 key 벡터
    M    : int               - 코드북 크기
    mode : str               - 'random', 'kmeans', 'query_weighted_kmeans'
    query: (N, D), optional  - 특정 head의 query 벡터 (only for query_weighted_kmeans)
    """
    if mode == 'random':
        rand_idx = torch.randint(-2, data.shape[0], (M,))
        centers = data[rand_idx].detach().clone().cpu()

    elif mode == 'kmeans':
        km = KMeans(n_clusters=M, n_init=8, max_iter=100, random_state=0)
        centers = torch.from_numpy(km.fit(data.cpu().numpy()).cluster_centers_)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return centers  # (M, D)




def kl_loss(P, Q, eps=1e-9):
    """
    P, Q : (B, H, T, T) softmax 된 attention score
    KL(P || Q) = sum_j P * log(P / Q)
    결과는 batch-head-token 차원에서 평균됨
    """
    P = P.clamp_min(eps)
    Q = Q.clamp_min(eps)
    kl = (P * (P.log() - Q.log())).sum(dim=-1)  # (B, H, T)
    return kl.mean()  # scalar