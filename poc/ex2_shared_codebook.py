

import traceback
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import os
import json


device = 'cuda'



def run_new_algorithm(data, K, num_iters=20, use_flash=True):
    """
    제안한 알고리즘: 각 데이터 x 를 m_k + α·m_l 로 근사
    data: (N, d) 텐서, K: 클러스터 수, num_iters: 반복 횟수
    각 데이터는 두 개의 인덱스 (k, l)로 할당됨.
    """
    N, d = data.shape
    # 초기 클러스터 중심: data에서 무작위 선택
    indices = torch.randperm(N)[:K]
    centers = data[indices].clone()  # (K, d)
    # α의 초기값 (스칼라)
    alpha = torch.tensor(0.01, device=device)
    # 각 데이터의 할당: assign_k (첫 번째 역할)와 assign_l (두 번째 역할)
    assign_k = torch.zeros(N, dtype=torch.long, device=device)
    assign_l = torch.zeros(N, dtype=torch.long, device=device)
    
    for it in range(num_iters):
        # --- 할당 단계 ---
        # 후보 근사: 각 (k, l) 쌍에 대해 candidate = centers[k] + alpha * centers[l]
        # candidate의 shape: (K, K, d)
        if use_flash:
            # O(K) greedy algorithm
            # 첫번째 역할(k) 찾기 - 데이터와 centers 간 거리 계산
            diff_k = data.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, d)
            errors_k = (diff_k ** 2).sum(dim=-1)  # (N, K)
            assign_k = torch.argmin(errors_k, dim=1)  # (N,)
            
            # 두번째 역할(l) 찾기 - 잔차와 centers 간 거리 계산
            residual = data - centers[assign_k]  # (N, d)
            diff_l = residual.unsqueeze(1) - (alpha * centers).unsqueeze(0)  # (N, K, d)
            errors_l = (diff_l ** 2).sum(dim=-1)  # (N, K)
            assign_l = torch.argmin(errors_l, dim=1)  # (N,)
            # k, l 인덱스를 하나의 코드로 변환
            min_indices = assign_k * K + assign_l
        else:
            candidate = centers.unsqueeze(1) + alpha * centers.unsqueeze(0)
            # 각 데이터 x (shape: (N, d))와 candidate 간의 거리 계산
            # data 확장: (N, 1, 1, d), candidate 확장: (1, K, K, d)
            diff = data.unsqueeze(1).unsqueeze(2) - candidate.unsqueeze(0)  # (N, K, K, d)
            errors = (diff ** 2).sum(dim=-1)  # (N, K, K)
            errors_flat = errors.view(N, -1)  # (N, K*K)
            min_indices = torch.argmin(errors_flat, dim=1)  # (N,)


        assign_k = min_indices // K  # 첫 번째 역할 index
        assign_l = min_indices % K   # 두 번째 역할 index
        
        # --- α 업데이트 ---
        num = 0.0
        den = 0.0
        # 각 데이터에 대해 할당된 (k, l) 쌍을 이용하여 α의 닫힌해 업데이트
        for n in range(N):
            k_idx = assign_k[n]
            l_idx = assign_l[n]
            num += torch.dot(centers[l_idx], data[n] - centers[k_idx])
            den += torch.dot(centers[l_idx], centers[l_idx])
        if den != 0:
            alpha = num / den
       

        # --- 클러스터 중심 업데이트 ---
        new_centers = centers.clone()
        for p in range(K):
            # p가 첫 번째 역할로 선택된 데이터의 인덱스 (A_p)
            idx_A = (assign_k == p).nonzero(as_tuple=False).squeeze()
            # p가 두 번째 역할로 선택된 데이터의 인덱스 (B_p)
            idx_B = (assign_l == p).nonzero(as_tuple=False).squeeze()
            
            term_num = torch.zeros(d, device=device)
            count_A = 0
            count_B = 0
            if idx_A.numel() > 0:
                if idx_A.dim() == 0:
                    idx_A = idx_A.unsqueeze(0)
                # A_p에 대해: x - alpha * centers[assign_l]
                term_num += (data[idx_A] - alpha * centers[assign_l[idx_A]]).sum(dim=0)
                count_A = idx_A.shape[0]
            if idx_B.numel() > 0:
                if idx_B.dim() == 0:
                    idx_B = idx_B.unsqueeze(0)
                # B_p에 대해: α * (x - centers[assign_k])
                term_num += alpha * (data[idx_B] - centers[assign_k[idx_B]]).sum(dim=0)
                count_B = idx_B.shape[0]
            denominator = count_A + (alpha ** 2) * count_B
            if denominator > 0:
                new_centers[p] = term_num / denominator
        centers = new_centers

    # --- MSE 계산 및 출력 ---
    approx = centers[assign_k] + alpha * centers[assign_l]
    squared_errors = (data - approx) ** 2
    mse = torch.mean(squared_errors)
    print("MSE:", mse.item())

    return centers, assign_k, assign_l, alpha


def run_new_algorithm_general(data, K, M, num_iters=20, use_flash=True):
    """
    제안한 알고리즘 일반화 (M개의 중심의 선형 결합)
    각 데이터 x 를 m_{a1} + alpha_2 * m_{a2} + ... + alpha_M * m_{aM} 으로 근사.
    
    data: (N, d) 텐서, K: 클러스터 수, M: 사용할 역할 수 (첫 번째 계수는 1로 고정)
    num_iters: 반복 횟수, use_flash: greedy 할당을 사용할지 여부.
    """
    import torch
    
    device = data.device
    N, d = data.shape
    
    # 초기 클러스터 중심: data에서 무작위 선택
    indices = torch.randperm(N)[:K]
    centers = data[indices].clone()  # (K, d)
    
    # alpha 벡터: 첫 번째 원소는 1 고정, 나머지 초기값은 0.5로 설정 (길이 M)
    alpha = torch.ones(M, device=device)
    if M > 1:
        alpha[1:] = 0.5
    
    # 각 데이터의 할당: assignments: (N, M) 텐서, 각 열은 해당 역할에 대한 클러스터 인덱스
    assignments = torch.zeros(N, M, dtype=torch.long, device=device)
    
    for it in range(num_iters):
        # --- 할당 단계 ---
        if use_flash:
            # 역할 1: data와 centers 간 거리로 할당 (첫 번째 역할)
            diff = data.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, d)
            errors = (diff ** 2).sum(dim=-1)  # (N, K)
            assignments[:, 0] = torch.argmin(errors, dim=1)
            
            # 이후 역할들: greedy 방식으로 잔차 업데이트
            residual = data - centers[assignments[:, 0]]  # 초기 residual
            for j in range(1, M):
                # 현재 역할 j+1: alpha[j]가 스칼라 계수
                # 후보: alpha[j] * centers
                diff_j = residual.unsqueeze(1) - (alpha[j] * centers).unsqueeze(0)  # (N, K, d)
                errors_j = (diff_j ** 2).sum(dim=-1)  # (N, K)
                assignments[:, j] = torch.argmin(errors_j, dim=1)
                # 잔차 업데이트: 현재 역할의 기여를 빼줌
                residual = residual - alpha[j] * centers[assignments[:, j]]
        else:
            # use_flash=False인 경우 모든 후보 조합(K^M)을 고려하는 것은 M>2일 때 비현실적이므로
            # 여기서는 use_flash를 권장합니다.
            raise NotImplementedError("Non-greedy candidate 방식은 M>2에서 구현하기 어렵습니다.")
        
        # --- α 업데이트 ---
        # alpha[0]는 1로 고정, 나머지 (M-1)개에 대해 정규방정식 A * alpha_rest = b
        if M > 1:
            A = torch.zeros(M-1, M-1, device=device)
            b = torch.zeros(M-1, device=device)
            for n in range(N):
                c1 = centers[assignments[n, 0]]
                for j in range(1, M):
                    cj = centers[assignments[n, j]]
                    b[j-1] += torch.dot(cj, data[n] - c1)
                    for k in range(1, M):
                        ck = centers[assignments[n, k]]
                        A[j-1, k-1] += torch.dot(cj, ck)
            # solve A * x = b, 만약 A가 singular 하면 최소제곱 해를 구함.
            # torch.linalg.solve는 A가 invertible 해야 하므로, pinv 사용 가능.
            if torch.det(A) != 0:
                alpha_rest = torch.linalg.solve(A, b)
            else:
                alpha_rest = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze()
            alpha[1:] = alpha_rest
        
        # --- 클러스터 중심 업데이트 ---
        new_centers = centers.clone()
        for p in range(K):
            # p가 역할 1 (첫 번째 역할)로 선택된 데이터 인덱스
            idx_role1 = (assignments[:, 0] == p).nonzero(as_tuple=False).squeeze()
            # 기여와 분모를 초기화
            term_num = torch.zeros(d, device=device)
            denom = 0.0
            
            # 역할 1 기여: x - (sum_{j=2}^M alpha[j]* centers[assignments[n, j]])
            if idx_role1.numel() > 0:
                if idx_role1.dim() == 0:
                    idx_role1 = idx_role1.unsqueeze(0)
                # 누적 sum: for 각 데이터 n, subtract 후 더함
                sub_term = torch.zeros((idx_role1.shape[0], d), device=device)
                for j in range(1, M):
                    sub_term += alpha[j] * centers[assignments[idx_role1, j]]
                term_num += (data[idx_role1] - sub_term).sum(dim=0)
                denom += idx_role1.numel()  # 역할1의 coefficient는 1
                
            # 역할 j (j>=2) 기여
            for j in range(1, M):
                idx_role = (assignments[:, j] == p).nonzero(as_tuple=False).squeeze()
                if idx_role.numel() > 0:
                    if idx_role.dim() == 0:
                        idx_role = idx_role.unsqueeze(0)
                    # 여기서, 다른 역할들의 기여는 centers[assignments[:,0]]와 j 이외의 역할
                    sub_term = torch.zeros((idx_role.shape[0], d), device=device)
                    # 첫 번째 역할 contribution
                    sub_term += centers[assignments[idx_role, 0]]
                    # 다른 역할들 (r != j, r>=2)
                    for r in range(1, M):
                        if r != j:
                            sub_term += alpha[r] * centers[assignments[idx_role, r]]
                    term_num += alpha[j] * (data[idx_role] - sub_term).sum(dim=0)
                    denom += (alpha[j]**2) * idx_role.numel()
            # 업데이트: denom이 0이 아니면, 새로운 중심 계산
            if denom > 0:
                new_centers[p] = term_num / denom
        centers = new_centers
        
    # --- MSE 계산 및 출력 ---
    # 재구성: 역할 1은 coefficient 1, 역할 j (j>=2)는 alpha[j]
    approx = centers[assignments[:, 0]].clone()
    for j in range(1, M):
        approx += alpha[j] * centers[assignments[:, j]]
    mse = ((data - approx) ** 2).mean()
    print("MSE:", mse.item())
    
    return centers, assignments, alpha





class ProductQuantizer:
    def __init__(self, M, Ks, random_state=42):
        """
        Product Quantizer 클래스.
        
        Parameters:
            M: int, 서브공간(서브양자화기)의 개수
            Ks: int, 각 서브공간에서 사용할 클러스터(코드북)의 개수
            random_state: 재현성을 위한 난수 시드
        """
        self.M = M
        self.Ks = Ks
        self.random_state = random_state
        self.centroids = None  # 각 서브공간의 클러스터 중심들을 저장할 리스트
        self.ds = None         # 각 서브공간의 차원 (전체 차원 d가 M으로 나누어 떨어져야 함)
        
    def fit(self, X):
        """
        주어진 데이터 X에 대해 product quantizer를 학습합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        d는 M으로 나누어 떨어져야 합니다.
        """
        n_samples, d = X.shape
        if d % self.M != 0:
        #    raise ValueError(f"데이터 차원 {d}는 서브공간 수 {self.M}로 나누어 떨어지지 않습니다.")
             res = d%self.M
             X = X[:-res]
             n_samples, d = X.shape
	
        self.ds = d // self.M
        self.centroids = []
        
        # 각 서브공간에 대해 KMeans 수행
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            kmeans = KMeans(n_clusters=self.Ks, random_state=self.random_state)
            kmeans.fit(sub_X)
            self.centroids.append(kmeans.cluster_centers_)
        return self
        
    def encode(self, X):
        """
        데이터 X를 product quantization 코드(각 서브공간의 인덱스)로 인코딩합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        Returns:
            codes: np.array, shape=(n_samples, M) – 각 행은 서브공간별 인덱스들을 포함합니다.
        """
        n_samples, d = X.shape
        if d != self.M * self.ds:
           # raise ValueError("데이터 차원이 학습 시와 일치하지 않습니다.")
           X = X[:, :self.M*self.ds]


        codes = np.empty((n_samples, self.M), dtype=np.int32)
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            centroids = self.centroids[m]
            # 각 서브벡터와 해당 서브공간의 모든 클러스터 중심 사이의 거리를 계산합니다.
            distances = np.linalg.norm(sub_X[:, None, :] - centroids[None, :, :], axis=2)
            codes[:, m] = np.argmin(distances, axis=1)
        return codes
    
    def encode_flash(self, X):
        return self.encode(X)


        
    def decode(self, codes):
        """
        인코딩된 코드를 원래 벡터에 근사하는 재구성 벡터로 복원합니다.
        
        Parameters:
            codes: np.array, shape=(n_samples, M)
            
        Returns:
            X_reconstructed: np.array, shape=(n_samples, d)
        """
        n_samples, M = codes.shape
        if M != self.M:
            raise ValueError("코드의 서브공간 수가 맞지 않습니다.")
        X_reconstructed = np.zeros((n_samples, self.M * self.ds))
        for m in range(self.M):
            centroids = self.centroids[m]
            X_reconstructed[:, m*self.ds:(m+1)*self.ds] = centroids[codes[:, m]]
        return X_reconstructed



class FixedAlphaProductQuantizer:
    def __init__(self, M, Ks, random_state=42):
        """
        Product Quantizer 클래스.
        
        Parameters:
            M: int, 서브공간(서브양자화기)의 개수
            Ks: int, 각 서브공간에서 사용할 클러스터(코드북)의 개수
            random_state: 재현성을 위한 난수 시드
        """
        self.M = M
        self.Ks = Ks
        self.random_state = random_state
        self.centroids = None  # 각 서브공간의 클러스터 중심들을 저장할 리스트
        self.ds = None         # 각 서브공간의 차원 (전체 차원 d가 M으로 나누어 떨어져야 함)
        
    def fit(self, X):
        """
        주어진 데이터 X에 대해 product quantizer를 학습합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        d는 M으로 나누어 떨어져야 합니다.
        """
        n_samples, d = X.shape
        if d % self.M != 0:
        #    raise ValueError(f"데이터 차원 {d}는 서브공간 수 {self.M}로 나누어 떨어지지 않습니다.")
             res = d%self.M
             X = X[:-res]
             n_samples, d = X.shape
	
        self.ds = d // self.M
        self.centroids = []
        
        # 각 서브공간에 대해 KMeans 수행
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            kmeans = KMeans(n_clusters=self.Ks, random_state=self.random_state)
            kmeans.fit(sub_X)
            self.centroids.append(kmeans.cluster_centers_)
        return self
        
    def encode(self, X):
        """
        데이터 X를 product quantization 코드(각 서브공간의 인덱스)로 인코딩합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        Returns:
            codes: np.array, shape=(n_samples, M) – 각 행은 서브공간별 인덱스들을 포함합니다.
        """
        n_samples, d = X.shape
        if d != self.M * self.ds:
           # raise ValueError("데이터 차원이 학습 시와 일치하지 않습니다.")
           X = X[:, :self.M*self.ds]


        codes = np.empty((n_samples, 2*self.M), dtype=np.int32)
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            centroids = self.centroids[m]
            # 각 서브벡터와 해당 서브공간의 모든 클러스터 중심 사이의 거리를 계산합니다.
            distances = np.linalg.norm(sub_X[:, None, :] - centroids[None, :, :], axis=2)
            codes[:, m] = np.argmin(distances, axis=1)

            rec = centroids[codes[:, m]]
            residual = sub_X - rec
 
            #print("old=", np.max(codes[:, m]))
            #print(np.max(codes[:, M+m]))
            # 각 서브벡터와 해당 서브공간의 모든 클러스터 중심 사이의 거리를 계산합니다.
            distances = np.linalg.norm(2*residual[:, None, :] - centroids[None, :, :], axis=2)
            codes[:, M+m] =  np.argmin(distances, axis=1)
            #print(np.max(codes[:, M+m]))

        return codes
    
    def encode_flash(self, X):
        return self.encode(X)


        
    def decode(self, codes):
        """
        인코딩된 코드를 원래 벡터에 근사하는 재구성 벡터로 복원합니다.
        
        Parameters:
            codes: np.array, shape=(n_samples, M)
            
        Returns:
            X_reconstructed: np.array, shape=(n_samples, d)
        """
        n_samples, M = codes.shape
        if M//2 != self.M:
            raise ValueError("코드의 서브공간 수가 맞지 않습니다.")
        X_reconstructed = np.zeros((n_samples, self.M * self.ds))
        for m in range(self.M):
            centroids = self.centroids[m]
            X_reconstructed[:, m*self.ds:(m+1)*self.ds] = centroids[codes[:, m]] + 0.5*centroids[codes[:,self.M+m]]

        return X_reconstructed

class AlphaProductQuantizer:
    def __init__(self, M, Ks, num_roles=2, random_state=42):
        """
        Product Quantizer 클래스.
        
        Parameters:
            M: int, 서브공간(서브양자화기)의 개수
            Ks: int, 각 서브공간에서 사용할 클러스터(코드북)의 개수
            random_state: 재현성을 위한 난수 시드
        """
        self.M = M
        self.Ks = Ks
        self.random_state = random_state
        self.centroids = None  # 각 서브공간의 클러스터 중심들을 저장할 리스트
        self.alphas = None
        self.ds = None         # 각 서브공간의 차원 (전체 차원 d가 M으로 나누어 떨어져야 함)
        self.use_new_way = False
        
    def fit(self, X):
        """
        주어진 데이터 X에 대해 product quantizer를 학습합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        d는 M으로 나누어 떨어져야 합니다.
        """
        X = torch.tensor(X,device='cuda')
        n_samples, d = X.shape
        if d % self.M != 0:
            raise ValueError(f"데이터 차원 {d}는 서브공간 수 {self.M}로 나누어 떨어지지 않습니다.")
        self.ds = d // self.M
        self.centroids = []
        self.alphas = []
        
        # 각 서브공간에 대해 KMeans 수행
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            centroids, assign_k , assign_l, alpha = run_new_algorithm(torch.tensor(sub_X).cuda(), self.Ks, use_flash=False)
            self.centroids.append(centroids)
            self.alphas.append(alpha)
            print("alpha=", alpha)
        return self
        
    def encode(self, X):
        """
        데이터 X를 product quantization 코드(각 서브공간의 인덱스)로 인코딩합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        Returns:
            codes: np.array, shape=(n_samples, M) – 각 행은 서브공간별 인덱스들을 포함합니다.
        """
        n_samples, d = X.shape
        if d != self.M * self.ds:
            raise ValueError("데이터 차원이 학습 시와 일치하지 않습니다.")
        codes = torch.empty((n_samples, self.M), dtype=torch.int32) #, dtype=np.int32)
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            centers = self.centroids[m]
            alpha = self.alphas[m]
            
            data = torch.tensor(sub_X, device='cuda')
            N, d = data.shape
            # --- 할당 단계 ---
            # 후보 근사: 각 (k, l) 쌍에 대해 candidate = centers[k] + alpha * centers[l]
            # candidate의 shape: (K, K, d)
            candidate = centers.unsqueeze(1) + alpha * centers.unsqueeze(0)
            # 각 데이터 x (shape: (N, d))와 candidate 간의 거리 계산
            # data 확장: (N, 1, 1, d), candidate 확장: (1, K, K, d)
            diff = data.unsqueeze(1).unsqueeze(2) - candidate.unsqueeze(0)  # (N, K, K, d)
            errors = (diff ** 2).sum(dim=-1)  # (N, K, K)
            errors_flat = errors.view(N, -1)  # (N, K*K)
            min_indices = torch.argmin(errors_flat, dim=1)  # (N,)
            assign_k = min_indices // self.Ks  # 첫 번째 역할 index
            assign_l = min_indices % self.Ks   # 두 번째 역할 index
            codes[:, m] = min_indices
        return codes
    
    def encode_flash(self, X):
        """
        O(K) greedy 알고리즘으로 인코딩을 수행합니다.
        alpha가 1보다 작다고 가정하여 첫번째 역할을 먼저 찾고, 두번째 역할을 찾습니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        Returns:
            codes: np.array, shape=(n_samples, M) - 각 행은 서브공간별 인덱스들을 포함합니다.
        """
        n_samples, d = X.shape
        if d != self.M * self.ds:
            raise ValueError("데이터 차원이 학습 시와 일치하지 않습니다.")
            
        codes = torch.empty((n_samples, self.M), dtype=torch.int32)
        
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            centers = self.centroids[m] 
            alpha = self.alphas[m]
            
            data = torch.tensor(sub_X, device='cuda')
            N, d = data.shape
            
            # 첫번째 역할(k) 찾기 - 데이터와 centers 간 거리 계산
            diff_k = data.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, d)
            errors_k = (diff_k ** 2).sum(dim=-1)  # (N, K)
            assign_k = torch.argmin(errors_k, dim=1)  # (N,)
            
            # 두번째 역할(l) 찾기 - 잔차와 centers 간 거리 계산
            residual = data - centers[assign_k]  # (N, d)
            diff_l = residual.unsqueeze(1) - (alpha * centers).unsqueeze(0)  # (N, K, d)
            errors_l = (diff_l ** 2).sum(dim=-1)  # (N, K)
            assign_l = torch.argmin(errors_l, dim=1)  # (N,)
            
            # k, l 인덱스를 하나의 코드로 변환
            codes[:, m] = assign_k * self.Ks + assign_l
            
        return codes


        
    def decode(self, codes):
        """
        인코딩된 코드를 원래 벡터에 근사하는 재구성 벡터로 복원합니다.
        
        Parameters:
            codes: np.array, shape=(n_samples, M)
            
        Returns:
            X_reconstructed: np.array, shape=(n_samples, d)
        """
        n_samples, M = codes.shape
        if M != self.M:
            raise ValueError("코드의 서브공간 수가 맞지 않습니다.")
        X_reconstructed = torch.zeros((n_samples, self.M * self.ds))
        for m in range(self.M):
            centroids = self.centroids[m]
            X_reconstructed[:, m*self.ds:(m+1)*self.ds] = centroids[codes[:, m]//self.Ks] + self.alphas[m]*centroids[codes[:, m]%self.Ks]

        return X_reconstructed.cpu().numpy()

#centroids, _ , alpha = run_new_algorithm_general(torch.tensor(sub_X).cuda(), self.Ks, self.L)


class GeneralAlphaProductQuantizer:
    def __init__(self, M, Ks, num_roles=2, random_state=42):
        """
        General Alpha Product Quantizer 클래스.
        
        Parameters:
            M: int, 서브공간(서브양자화기)의 개수
            Ks: int, 각 서브공간에서 사용할 클러스터(코드북)의 개수
            num_roles: 각 서브공간에서 사용할 역할(선형 결합에 참여하는 중심)의 수 (첫 번째 역할의 계수는 1로 고정)
            random_state: 재현성을 위한 난수 시드
        """
        self.M = M              # 서브공간 수
        self.Ks = Ks            # 각 서브공간의 클러스터 수
        self.num_roles = num_roles  # 각 서브공간의 역할 수 (기존 M과는 별도)
        self.random_state = random_state
        self.centroids = []     # 각 서브공간별 클러스터 중심 (Tensor, shape: (Ks, d_sub))
        self.alphas = []        # 각 서브공간별 alpha 벡터 (Tensor, shape: (num_roles, ), 첫 번째 원소는 1)
        self.ds = None          # 각 서브공간의 차원 (전체 차원 d가 M으로 나누어 떨어져야 함)
        
    def fit(self, X):
        """
        주어진 데이터 X에 대해 product quantizer를 학습합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
        
        d는 self.M으로 나누어 떨어져야 합니다.
        """
        X_tensor = torch.tensor(X, device='cuda')
        n_samples, d = X_tensor.shape
        if d % self.M != 0:
            raise ValueError(f"데이터 차원 {d}는 서브공간 수 {self.M}로 나누어 떨어지지 않습니다.")
        self.ds = d // self.M
        self.centroids = []
        self.alphas = []
        
        # 각 서브공간에 대해 run_new_algorithm_general 수행
        for m in range(self.M):
            sub_X = X_tensor[:, m*self.ds:(m+1)*self.ds]
            centers, assignments, alpha = run_new_algorithm_general(sub_X, self.Ks, self.num_roles)
            self.centroids.append(centers)
            self.alphas.append(alpha)
            print(f"Subspace {m}: alpha =", alpha.cpu().numpy())
        return self

    def encode(self, X):
        """
        일반적인 방식으로 인코딩을 수행합니다.
        모든 후보 조합(K^(num_roles))을 고려하여 인덱스를 결정합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        Returns:
            codes: np.array, shape=(n_samples, M) – 각 행은 서브공간별 단일 인덱스 (0 ~ Ks^(num_roles)-1)를 포함합니다.
        """
        if self.num_roles > 2:
            print("Warning: Using encode() with num_roles > 2 is inefficient. Falling back to encode_flash().")
            return self.encode_flash(X)
        
        n_samples, d = X.shape
        if d != self.M * self.ds:
            raise ValueError("데이터 차원이 학습 시와 일치하지 않습니다.")
            
        codes = torch.empty((n_samples, self.M), dtype=torch.long, device='cuda')
        
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            centers = self.centroids[m]    # (Ks, d_sub)
            alpha = self.alphas[m]         # (num_roles,)
            
            data = torch.tensor(sub_X, device='cuda')
            N, d_sub = data.shape
            # 모든 후보 조합에 대해 candidate 계산 (비효율적이므로 num_roles>2에서는 use_flash 권장)
            # candidate shape: (Ks^num_roles, d_sub)
            # 먼저, 모든 역할의 인덱스 조합을 구합니다.
            grid = torch.stack(torch.meshgrid(*[torch.arange(self.Ks, device='cuda') for _ in range(self.num_roles)], indexing='ij'), dim=-1)
            grid = grid.view(-1, self.num_roles)  # (Ks^(num_roles), num_roles)
            # 각 조합에 대해 재구성 벡터: centers[grid[:,0]] + sum_{r=1}^{num_roles-1} alpha[r] * centers[grid[:,r]]
            candidate = centers[grid[:, 0]].clone()
            for r in range(1, self.num_roles):
                candidate += alpha[r] * centers[grid[:, r]]
            # data: (N, d_sub), candidate: (Ks^(num_roles), d_sub)
            diff = data.unsqueeze(1) - candidate.unsqueeze(0)  # (N, Ks^(num_roles), d_sub)
            errors = (diff ** 2).sum(dim=-1)  # (N, Ks^(num_roles))
            min_indices = torch.argmin(errors, dim=1)  # (N,)
            codes[:, m] = min_indices
        return codes.cpu().numpy()
    
    def encode_flash(self, X):
        """
        Greedy 알고리즘을 이용한 인코딩.
        각 서브공간에 대해 첫 번째 역할부터 순차적으로 선택합니다.
        
        Parameters:
            X: np.array, shape=(n_samples, d)
            
        Returns:
            codes: np.array, shape=(n_samples, M) – 각 행은 서브공간별 단일 인덱스 (0 ~ Ks^(num_roles)-1)를 포함합니다.
        """
        n_samples, d = X.shape
        if d != self.M * self.ds:
            raise ValueError("데이터 차원이 학습 시와 일치하지 않습니다.")
            
        codes = torch.empty((n_samples, self.M), dtype=torch.long, device='cuda')
        
        for m in range(self.M):
            sub_X = X[:, m*self.ds:(m+1)*self.ds]
            centers = self.centroids[m]  # (Ks, d_sub)
            alpha = self.alphas[m]       # (num_roles,)
            
            data = torch.tensor(sub_X, device='cuda')
            N, d_sub = data.shape
            # assignments: (N, num_roles)
            assignments = torch.empty((N, self.num_roles), dtype=torch.long, device='cuda')
            # 첫 번째 역할 선택
            diff = data.unsqueeze(1) - centers.unsqueeze(0)  # (N, Ks, d_sub)
            errors = (diff ** 2).sum(dim=-1)  # (N, Ks)
            assignments[:, 0] = torch.argmin(errors, dim=1)
            
            residual = data - centers[assignments[:, 0]]
            # 이후 역할들
            for r in range(1, self.num_roles):
                diff_r = residual.unsqueeze(1) - (alpha[r] * centers).unsqueeze(0)  # (N, Ks, d_sub)
                errors_r = (diff_r ** 2).sum(dim=-1)  # (N, Ks)
                assignments[:, r] = torch.argmin(errors_r, dim=1)
                residual = residual - alpha[r] * centers[assignments[:, r]]
            # 인코딩: 각 데이터에 대해 assignments를 하나의 정수로 결합 (base = Ks)
            codes_sub = torch.zeros(N, dtype=torch.long, device='cuda')
            for r in range(self.num_roles):
                codes_sub = codes_sub * self.Ks + assignments[:, r]
            codes[:, m] = codes_sub
            
        return codes.cpu().numpy()
    
    def decode(self, codes):
        """
        인코딩된 코드를 원래 벡터에 근사하는 재구성 벡터로 복원합니다.
        
        Parameters:
            codes: np.array, shape=(n_samples, M) – 각 값은 0 ~ Ks^(num_roles)-1 사이의 정수.
            
        Returns:
            X_reconstructed: np.array, shape=(n_samples, d)
        """
        codes_tensor = torch.tensor(codes, device='cuda', dtype=torch.long)
        n_samples, num_subspaces = codes_tensor.shape
        if num_subspaces != self.M:
            raise ValueError("코드의 서브공간 수가 맞지 않습니다.")
        X_reconstructed = torch.empty((n_samples, self.M * self.ds), device='cuda')
        
        for m in range(self.M):
            centers = self.centroids[m]  # (Ks, d_sub)
            alpha = self.alphas[m]       # (num_roles,)
            codes_sub = codes_tensor[:, m]  # (N,)
            # codes_sub는 하나의 정수로 결합되어 있으므로, 각 역할의 인덱스로 분리합니다.
            assignments = torch.empty((n_samples, self.num_roles), dtype=torch.long, device='cuda')
            tmp = codes_sub.clone()
            for r in reversed(range(self.num_roles)):
                assignments[:, r] = tmp % self.Ks
                tmp = tmp // self.Ks
            # 재구성: centers[assignments[:,0]] + sum_{r=1}^{num_roles-1} alpha[r]*centers[assignments[:,r]]
            recon = centers[assignments[:, 0]].clone()
            for r in range(1, self.num_roles):
                recon = recon + alpha[r] * centers[assignments[:, r]]
            X_reconstructed[:, m*self.ds:(m+1)*self.ds] = recon
        return X_reconstructed.cpu().numpy()




# 예제 사용법
if __name__ == "__main__":
    # 무작위 데이터 생성 (예: 1000개 샘플, 64차원)
    np.random.seed(42)
    #n_samples = 1000
    #d = 64
    #X = np.random.randn(n_samples, d)


    # 결과 저장 디렉토리 경로
    results_dir = "./quantization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 현재 날짜 및 시간으로 파일 이름 생성
    
    # 현재 날짜 및 시간을 포맷팅하여 파일 이름 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"result_{current_time}.json"


    for bit_per_element in [1]: #[0.5, 1, 2]:
        for Ks in [256, 512]:
        #for group_size in [4, 8, 16, 32, 64]:
            for num_roles in [1,2]: #, 2]: #, 3]: #, 3, 4]:
                X = torch.load("/hai/projects/rqllm/poc/datasets_kv/wikitext2_train_0p1_opt125m_keys.pt").numpy()
                X_org = X
                X = X_org[:10000]
                X_test = X_org[20000:30000]
                n_channels = X.shape[1]
			#group_size = 4
                #bit_per_element = 1
                #num_roles = 1
		
                #if num_roles==1:
                group_size = int(np.log2(Ks)*num_roles/bit_per_element) # 각 서브공간당 클러스터 수 (256개)
                #else:
                #     group_size = 64
                #     num_roles = int(group_size*bit_per_element/np.log2(Ks))
                #Ks = 2**(int(group_size*bit_per_element)//num_roles)
                M = n_channels // group_size

                # Print all parameters
                print("\n=== Product Quantization Parameters ===")
                print(f"Number of samples: {X.shape[0]}")
                print(f"Number of channels: {n_channels}")
                print(f"Group size: {group_size}")
                print(f"Number of subspaces (M): {M}")
                print(f"Bits per element: {bit_per_element}")
                print(f"Number of roles: {num_roles}")
                print(f"Clusters per subspace (Ks): {Ks}")
                print("=======================================\n")
    

                print(Ks)

                error_message = ""
                try:
                    if num_roles==1:
                        pq = ProductQuantizer(M, Ks)
                    elif num_roles==2:
                        pq = FixedAlphaProductQuantizer(M, Ks)
                    else:
                        pq = GeneralAlphaProductQuantizer(M, Ks, num_roles=num_roles)
 
                    pq.fit(X)
                    
                    print(X.shape)
                    print(X_test.shape)
                    # 데이터 인코딩 (압축)
                    if X_test.shape[1]!=pq.M * pq.ds:
                        X_test[:, :pq.M*pq.ds]
                    codes = pq.encode(X_test)
                    codes_flash = pq.encode_flash(X_test)

                    # 인코딩된 코드를 통해 데이터 재구성
                    X_reconstructed = pq.decode(codes)
                    X_reconstructed_flash = pq.decode(codes_flash)

                    if X_test.shape[1]!=X_reconstructed.shape[1]:
                         X_test = X_test[:,:X_reconstructed.shape[1]]

                    # 재구성 오차 (평균제곱오차) 계산
                    mse = np.mean((X_test - X_reconstructed) ** 2)
                    print("평균 제곱 재구성 오차 (MSE):", mse)
                    mse_flash = np.mean((X_test - X_reconstructed_flash) ** 2)
                    print("평균 제곱 재구성 오차 (MSE):", mse_flash)

                except Exception as e:
                    traceback.print_exc()
                    print(f"Error: {e}")
                    error_message = str(e)
                    continue
                               # 결과를 저장할 딕셔너리 생성
                result = {
                    'config': {
                        'group_size': group_size,
                        'bit_per_element': bit_per_element,
                        'num_roles': num_roles,
                        'n_channels': n_channels,
                        'M': M,
                        'Ks': Ks
                    },
                    'metrics': {
                        'mse': float(mse),
                        'mse_flash': float(mse_flash),
                        'error': error_message
                    }
                }
                # 이전 결과 파일이 있는지 확인하고 결과를 추가
                
                
                # 모든 결과를 저장할 파일 경로
                all_results_file = os.path.join(results_dir, result_filename)
                
                # 기존 결과 파일이 있으면 불러오고, 없으면 새로운 리스트 생성
                if os.path.exists(all_results_file):
                    with open(all_results_file, 'r') as f:
                        try:
                            all_results = json.load(f)
                        except json.JSONDecodeError:
                            all_results = []
                else:
                    all_results = []
                
                # 현재 결과를 추가
                all_results.append(result)
                
                # 모든 결과를 파일에 저장
                with open(all_results_file, 'w+') as f:
                    json.dump(all_results, f, indent=4)
                
                print(f"결과가 {all_results_file}에 추가되었습니다.")




