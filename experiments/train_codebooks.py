


from sklearn.cluster import KMeans
import torch, torch.nn.functional as F
from types import SimpleNamespace
import argparse
import sys
import datetime
from rqllm.utils import *
from rqllm.qwk import query_weighted_kmeans
from torch import nn
from rqllm.datasets import create_dataset
from rqllm.pq import *
from rqllm.rq import *


torch.manual_seed(0)




#M =  256
#L = 1
#LR = 1e-2
#EPOCH = 1000

#Q_, K_, V_, meta = create_dataset('random')
#codebooks_init = init_codebooks_rq(K_, meta, M, L, init='kmeans')
#codebooks = train_codebook_core_rq(codebooks_init, Q_, K_, V_, meta, LR=LR, EPOCH=EPOCH, TAU=0.1)

#train_codebook_rq(dataset='random', M=M, L=L, LR=LR, EPOCH=EPOCH, TAU=0.1, init='kmeans')
# =======================================================
# 1. K-Means 기반 다단계 초기 코드북
# =======================================================


# ---------------- 4. 최종 평가 ----------------
"""
print("\n===== [FINAL EVALUATION] =====")
with torch.no_grad():
    residual = K; K_hat_total = 0
    for CB in codebooks:
        K_stage_hat, _, _ = soft_quantize(residual, CB, tau=TAU_MIN, ste=True)
        K_hat_total += K_stage_hat
        residual = residual - K_stage_hat

    P_hat,_ = attention(Q, K_hat_total, V)
    final_kl   = kl_loss(P_ref, P_hat).item()
    final_amse = F.mse_loss(P_hat, P_ref).item()
    final_kmse = F.mse_loss(K_hat_total, K).item()

    print(f"Final KL      : {final_kl:.3e}")
    print(f"Final Att-MSE : {final_amse:.3e}")
    print(f"Final Key-MSE : {final_kmse:.3e}")
"""


def parse_cli():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',  choices=['pq', 'rq', 'qwk'], default='rq', help='Training algorithm')
    p.add_argument('--dataset',type=str,  default='random8', help='Dataset')
    p.add_argument('--M',     type=int, default=256,       help='Codebook size')
    p.add_argument('--n_subvec', type=int, default=2,      help='PQ: number of sub‑vectors')
    p.add_argument('--L',     type=int, default=2,         help='RQ: number of stages')
    p.add_argument('--lr',    type=float, default=1e-2,    help='Learning rate')
    p.add_argument('--epoch', type=int, default=1000,      help='Number of epochs')
    p.add_argument('--tau',   type=float, default=0.1,     help='Softmax temperature')
    p.add_argument('--seed',  type=int, default=0,         help='Random seed')
    p.add_argument('--init',  type=str, default='kmeans',  help='Initialization method')
    return p.parse_args()


def main():
    args = parse_cli()

    #M =  256
    #L = 1
    #LR = 1e-2
    #EPOCH = 1000

    #Q_, K_, V_, meta = create_dataset('random')
    #codebooks_init = init_codebooks_rq(K_, meta, M, L, init='kmeans')
    #codebooks = train_codebook_core_rq(codebooks_init, Q_, K_, V_, meta, LR=LR, EPOCH=EPOCH, TAU=0.1)

    #train_codebook_rq(dataset='random', M=M, L=L, LR=LR, EPOCH=EPOCH, TAU=0.1, init='kmeans')
#
    if args.mode == 'pq':
        train_codebook_pq(dataset=args.dataset, EPOCH=args.epoch, n_subvec=args.n_subvec, M=args.M, init=args.init, LR=args.lr, TAU=args.tau)
    elif args.mode == 'rq':
        train_codebook_rq(dataset=args.dataset, EPOCH=args.epoch, L=args.L, M=args.M, init=args.init, LR=args.lr, TAU= args.tau)
    elif args.mode == 'qwk':
        Q_, K_, V_, meta, mask = create_dataset(args.dataset)
        query_weighted_kmeans(Q_, K_, V_, meta, M=args.M)

    # ----- final summary -----
    cmdline = 'python ' + ' '.join(sys.argv)
    print('\n===== [TRAINING COMPLETE] =====')
    print(f'Finished at {datetime.datetime.now().strftime("%Y‑%m‑%d %H:%M:%S")}')
    print(f'Command  : {cmdline}')


if __name__ == '__main__':
    main()



