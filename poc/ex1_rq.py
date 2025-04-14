import torch

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------
# Synthetic dataset generator
# ---------------------------
def generate_data(n_samples=10000, dim=64, random_state=42):
    np.random.seed(random_state)
    return np.random.randn(n_samples, dim)

# -------------------------------------
# Product Quantization (PQ) implementation
# -------------------------------------
class ProductQuantizer:
    def __init__(self, M, k, random_state=42):
        """
        Parameters:
          M: Number of subspaces (and hence number of separate quantizers).
          k: Number of clusters per subspace (each sub-quantizer outputs an index with log2(k) bits).
        """
        self.M = M
        self.k = k
        self.sub_quantizers = []
        self.d_sub = None
        self.random_state = random_state

    def fit(self, X):
        n, d = X.shape
        if d % self.M != 0:
            raise ValueError("The dimension must be divisible by the number of subspaces M")
        self.d_sub = d // self.M
        self.sub_quantizers = []

        # Train a quantizer for each subspace
        for m in range(self.M):
            X_sub = X[:, m*self.d_sub:(m+1)*self.d_sub]
            kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_sub)
            self.sub_quantizers.append(kmeans)

    def encode(self, X):
        """Encode the dataset into quantization indices for each subspace."""
        codes = []
        for m, quantizer in enumerate(self.sub_quantizers):
            X_sub = X[:, m*self.d_sub:(m+1)*self.d_sub]
            codes.append(quantizer.predict(X_sub))
        # Shape: (n_samples, M)
        return np.stack(codes, axis=1)

    def decode(self, codes):
        """Reconstruct the approximate vectors from the quantized codes."""
        n = codes.shape[0]
        X_rec = np.zeros((n, self.M * self.d_sub))
        for m, quantizer in enumerate(self.sub_quantizers):
            centroids = quantizer.cluster_centers_
            # codes for subspace m
            codes_sub = codes[:, m]
            X_rec[:, m*self.d_sub:(m+1)*self.d_sub] = centroids[codes_sub]
        return X_rec

    def distortion(self, X):
        codes = self.encode(X)
        X_rec = self.decode(codes)
        return mean_squared_error(X, X_rec)

# -------------------------------------
# Residual Quantization (RQ) implementation
# -------------------------------------
class ResidualQuantizer:
    def __init__(self, T, k, random_state=42):
        """
        Parameters:
          T: Number of quantization stages.
          k: Number of clusters per stage.
        """
        self.T = T
        self.k = k
        self.quantizers = []
        self.random_state = random_state

    def fit(self, X):
        n, d = X.shape
        self.quantizers = []
        residual = X.copy()
        # Train T stages of quantization on the residuals
        for t in range(self.T):
            kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
            kmeans.fit(residual)
            self.quantizers.append(kmeans)
            # Update the residual
            residual = residual - kmeans.cluster_centers_[kmeans.labels_]

    def encode(self, X):
        """Encode X into a sequence of indices for each stage."""
        n, d = X.shape
        codes = np.empty((n, self.T), dtype=np.int32)
        residual = X.copy()
        for t, quantizer in enumerate(self.quantizers):
            codes[:, t] = quantizer.predict(residual)
            residual = residual - quantizer.cluster_centers_[codes[:, t]]
        return codes

    def decode(self, codes):
        """Reconstruct the vectors from the codes by summing the stage centroids."""
        n = codes.shape[0]
        d = self.quantizers[0].cluster_centers_.shape[1]
        X_rec = np.zeros((n, d))
        for t, quantizer in enumerate(self.quantizers):
            centroids = quantizer.cluster_centers_
            codes_t = codes[:, t]
            X_rec += centroids[codes_t]
        return X_rec

    def distortion(self, X):
        codes = self.encode(X)
        X_rec = self.decode(codes)
        return mean_squared_error(X, X_rec)

class SingleCodebookResidualQuantizer:
    def __init__(self, T, k, random_state=42):
        """
        Parameters:
          T: Number of quantization stages.
          k: Number of clusters per stage.
        """
        self.T = T
        self.k = k
        self.quantizers = []
        self.random_state = random_state

    def fit(self, X):
        n, d = X.shape
        self.quantizers = []
        residual = X.copy()
        # Train T stages of quantization on the residuals
        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
        kmeans.fit(residual)

        for t in range(self.T):
            self.quantizers.append(kmeans)
            # Update the residual
            #residual = residual - kmeans.cluster_centers_[kmeans.labels_]

    def encode(self, X, cst):
        """Encode X into a sequence of indices for each stage."""
        n, d = X.shape
        codes = np.empty((n, self.T), dtype=np.int32)
        residual = X.copy()
        alpha = 1.0
        for t, quantizer in enumerate(self.quantizers):
            codes[:, t] = quantizer.predict(alpha*residual)
            residual = residual - (1.0/alpha)*quantizer.cluster_centers_[codes[:, t]]
            alpha *= cst
        return codes

    def decode(self, codes, cst):
        """Reconstruct the vectors from the codes by summing the stage centroids."""
        n = codes.shape[0]
        d = self.quantizers[0].cluster_centers_.shape[1]
        X_rec = np.zeros((n, d))
        alpha = 1.0
        for t, quantizer in enumerate(self.quantizers):
            centroids = quantizer.cluster_centers_
            codes_t = codes[:, t]
            X_rec += alpha*centroids[codes_t]
            alpha *= (1/cst)
        return X_rec

    def distortion(self, X, alpha):
        codes = self.encode(X, alpha)
        X_rec = self.decode(codes, alpha)
        return mean_squared_error(X, X_rec)

#--------------------------
# query-depending pruning
#---------------------------

def compute_max_dot_per_stage(q, codebooks):
    """
    Given a query vector q and a list of codebooks,
    compute for each stage t the maximum dot product:
      M[t] = max_j ( q^T Q_t(j) )
    where Q_t(j) is the j-th centroid in the t-th codebook.
    """
    M = []
    for cb in codebooks:
        # Each codebook is assumed to be an array of shape (k, d)
        # Compute dot product between q and all centroids
        dots = np.dot(cb, q)  # shape: (k,)
        M.append(np.max(dots))
    return M


def compute_min_dot_per_stage(q, codebooks):
    """
    Given a query vector q and a list of codebooks,
    compute for each stage t the maximum dot product:
      M[t] = max_j ( q^T Q_t(j) )
    where Q_t(j) is the j-th centroid in the t-th codebook.
    """
    M = []
    for cb in codebooks:
        # Each codebook is assumed to be an array of shape (k, d)
        # Compute dot product between q and all centroids
        dots = np.dot(cb, q)  # shape: (k,)
        M.append(np.min(dots))
    return M

def partial_dot_product(q, codes, codebooks, t):
    """
    Compute the partial dot product similarity using the first t stages.
    
    q: query vector (d-dimensional).
    codes: candidate's list of indices for each stage; length T.
    codebooks: list of codebooks.
    t: stage up to which to sum.
    
    Returns the partial similarity S_t = sum_{i=0}^{t-1} q^T Q_i(codes[i])
    """
    #S_t = 0.0
    #for i in range(t):
    #    centroid = codebooks[i][codes[i]]  # get the centroid (d-dimensional)
    #    S_t += np.dot(q, centroid)
    centroid = codebooks[t-1][codes[t-1]]  # get the centroid (d-dimensional)
    return np.dot(q, centroid)



def selective_lookup_dot(q, candidates_codes, codebooks, dont_prune=False):
    """
    Perform selective lookup on candidates given their residual quantization codes.
    Similarity is computed by a dot product between the query and the candidate's
    full reconstruction. The reconstruction is given by summing stage centroids.
    
    Parameters:
      q: (d,) array representing the query vector.
      candidates_codes: list of candidate codes; each is a list or array of T stage indices.
      codebooks: list of T codebooks (each of shape (k, d)).
      
    Returns:
      best_candidate: index in candidates_codes with highest dot product.
      best_score: its dot product score.
      scores: the full dot product scores for all candidates that were fully evaluated.
    """
    T = len(codebooks)
    # Precompute maximum possible dot contributions for each stage given q.
    max_dots = compute_max_dot_per_stage(q, codebooks)
    # For each possible stage t, compute the sum of max contributions from remaining stages.
    # That is: R_t = sum_{i=t}^{T-1} max_dots[i]
    R = np.zeros(T + 1)
    for t in range(T-1, -1, -1):
        R[t] = R[t+1] + max_dots[t]
    
    best_candidate = None
    best_score = -np.inf
    scores = np.full(len(candidates_codes), -np.inf)

    n_dot_products = 0
    # Process each candidate
    for idx, codes in enumerate(candidates_codes):
        pruned = False
        S_t = 0.0
        # Process stages one by one
        for t in range(1, T + 1):
            S_t += partial_dot_product(q, codes, codebooks, t)
            n_dot_products += 1
            # Compute optimistic upper bound (partial sum plus maximum possible remaining)
            upper_bound = S_t + R[t]
            if not dont_prune and upper_bound <= best_score:
                # Candidate cannot beat the current best
                pruned = True
                break
        
        if not pruned:
            # Compute full dot product if the candidate wasn't pruned
            #full_score = 0
            #for t in range(1, T+1):
            #    full_score += partial_dot_product(q, codes, codebooks, t)
            full_score = S_t
            scores[idx] = full_score
            if full_score > best_score:
                best_score = full_score
                best_candidate = idx
    
    print(f"Number of dot products: {n_dot_products}")    

    return best_candidate, best_score, scores


#--------------------------
# query-independent pruning
#---------------------------


def precompute_codebook_bounds(codebooks):
    """
    Precompute the maximum centroid norm for each codebook.
    This is query-independent.
    
    Parameters:
      codebooks: list of codebooks (each of shape (k, d)).
      
    Returns:
      max_norms: list of maximum centroid norms for each codebook.
    """
    max_norms = []
    for cb in codebooks:
        # Compute norm for each centroid in the codebook.
        norms = np.linalg.norm(cb, axis=1)
        max_norms.append(np.max(norms))
    return max_norms

def compute_remaining_bound(query, max_norms, stage):
    """
    Compute the remaining bound from stage 'stage' to the last stage,
    given the query and precomputed maximum centroid norms.
    
    Parameters:
      query: query vector (d,).
      max_norms: list of max centroid norms per stage.
      stage: current stage (0-indexed; e.g., stage=1 means already processed stage 0).
      
    Returns:
      bound: sum_{i=stage}^{T-1} ||q|| * max_norms[i]
    """
    q_norm = np.linalg.norm(query)
    remaining_bound = sum(q_norm * m for m in max_norms[stage:])
    return remaining_bound

def selective_lookup_dot_with_precomputed_bound(q, candidates_codes, codebooks, max_norms):
    """
    Similar to our previous selective lookup but using a query-independent bound.
    
    Parameters:
      q: query vector.
      candidates_codes: list of candidate codes (each is list/array of T indices).
      codebooks: list of codebooks for each stage.
      max_norms: precomputed list of maximum centroid norms per codebook.
      
    Returns:
      best_candidate: candidate with highest computed dot product.
      best_score: its dot product.
    """
    T = len(codebooks)
    best_candidate = None
    best_score = -np.inf

    #def partial_dot_product(q, codes, codebooks, t):
    #    S = 0.0
    #    for i in range(t):
    #        S += np.dot(q, codebooks[i][codes[i]])
    #    return S

    n_dot_products = 0
    for idx, codes in enumerate(candidates_codes):
        pruned = False
        S_t = 0.0
        # Process stage by stage
        for t in range(1, T + 1):
            S_t += partial_dot_product(q, codes, codebooks, t)
            n_dot_products += 1
            # Get query-independent remaining bound starting from stage t
            remaining_bound = compute_remaining_bound(q, max_norms, t)
            upper_bound = S_t + remaining_bound
            if upper_bound <= best_score:
                pruned = True
                break

        if not pruned:
            #full_score = partial_dot_product(q, codes, codebooks, T)
            full_score = S_t
            #full_score = 0
            #for t in range(1, T+1):
            #    full_score += partial_dot_product(q, codes, codebooks, t)
            if full_score > best_score:
                best_score = full_score
                best_candidate = idx

    print(f"Number of dot products: {n_dot_products}")    
    return best_candidate, best_score

def compute_attention_scores_by_pruning(q, candidates_codes, codebooks, theta=0.01, m=-1.0):
    """
    query_value : 관심 query의 z_i 값
    processed_keys : 지금까지 순차적으로 들어온 다른 key들의 z_j 값 리스트 (query 제외)
    total_keys : 전체 key의 개수
    m : 다른 key들의 최소값 가정 (즉, 모든 미처리 key에 대해 z_j >= m)
    """
    # Precompute maximum possible dot contributions for each stage given q.
    T = len(codebooks)
    max_dots = compute_max_dot_per_stage(q, codebooks)
    min_dots = compute_min_dot_per_stage(q, codebooks)

    # For each possible stage t, compute the sum of max contributions from remaining stages.
    # That is: R_t = sum_{i=t}^{T-1} max_dots[i]
    R = np.zeros(T + 1)
    for t in range(T-1, -1, -1):
        R[t] = R[t+1] + max_dots[t]

    Rm = np.zeros(T + 1)
    for t in range(T-1, -1, -1):
        Rm[t] = Rm[t+1] + min_dots[t]  

         # query의 지수 값
    query_exp = np.exp(q)   
    scores = np.zeros(len(candidates_codes))
    best_score = -np.inf
    N = len(candidates_codes)
    processed_sum = 0.0
    n_dot_products = 0
    remain = N

    for idx, codes in enumerate(candidates_codes):
        denominator_lower_bound = processed_sum + remain * np.exp(m)
        upper_bound = query_exp / denominator_lower_bound

        S_t = 0.0
        for t in range(1, T + 1):
            S_t += partial_dot_product(q, codes, codebooks, t)
            n_dot_products += 1
            scores[idx] = S_t
            z_upper_bound = S_t + R[t]
            z_lower_bound = S_t + Rm[t]
            assert z_lower_bound<=z_upper_bound
            e_z = np.exp(z_upper_bound)/(np.exp(z_lower_bound)+denominator_lower_bound)
            if e_z<=theta:
                break

        processed_sum += np.exp(scores[idx])
        remain = N - idx - 1

    print(f"Number of dot products: {n_dot_products}")
    scores = softmax(scores/np.sqrt(q.shape[0]))
    return scores


    return upper_bound

def softmax(x):

    """
    입력 벡터 x에 대해 softmax 값을 반환합니다.
    x는 1차원 혹은 다차원 배열일 수 있으며, 마지막 축을 따라 softmax 연산을 수행합니다.
    """
    # 수치적 안정성을 위해 최대값을 빼줍니다.
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def compute_attention_score(q, candidates_codes, codebooks):
    T = len(codebooks)
    # Precompute maximum possible dot contributions for each stage given q.
  
    scores = np.zeros(len(candidates_codes))
    for idx, codes in enumerate(candidates_codes):
        S_t = 0.0
        for t in range(1, T + 1):
            S_t += partial_dot_product(q, codes, codebooks, t)
        scores[idx] = S_t
    
    scores = softmax(scores/np.sqrt(q.shape[0]))
    return scores





# ---------------------------
# Main routine to compare PQ and RQ
# ---------------------------
def main():
    #
    # Parameters for synthetic data and quantizers
    np.random.seed(42)  # Set random seed for reproducibility
    n_samples = 5000
    dim = 64
    #X = generate_data(n_samples, dim)
    data = torch.load("/hai/projects/rqllm/poc/datasets_kv/wikitext2_train_0p1_opt125m_keys.pt").numpy()
    X = data[:10000]

    # Choose the same overall compression rate.
    # For PQ: each vector is encoded by M indices; each index requires log2(k) bits.
    # For RQ: each vector is encoded by T indices; hence, to have the same bits per vector, set M = T.
    M = 8   # Number of subspaces for PQ (compression rate = M * log2(k))
    T = M   # Number of stages for RQ; same as M to have equal rate.
    k = 256 # For example, 256 clusters means each index is 8 bits.

    # Train and compute distortions for PQ
    #pq = ProductQuantizer(M=M, k=k)
    #print("Fitting Product Quantizer ...")
    #pq.fit(X)
    #pq_dist = pq.distortion(X)
    #print("PQ distortion (MSE): {:.6f}".format(pq_dist))
    
    # Train and compute distortions for RQ
    rq = ResidualQuantizer(T=T, k=k)
    print("Fitting Residual Quantizer ...")
    rq.fit(X)
    rq_dist = rq.distortion(X)
    print("RQ distortion (MSE): {:.6f}".format(rq_dist))


    codebooks = [q.cluster_centers_ for q in rq.quantizers]
    
    # Simulate candidates: each candidate is represented by T indices
    candidates_codes = rq.encode(X)
    
    # Generate a random query vector
    q = np.random.randn(dim)
    
    # Apply selective lookup based on dot product upper bounds
    best_candidate, best_score, scores = selective_lookup_dot(q, candidates_codes, codebooks, dont_prune=True)
    print("Best candidate index:", best_candidate)
    print("Best dot product score:", best_score)

    best_candidate, best_score, scores = selective_lookup_dot(q, candidates_codes, codebooks, dont_prune=False)
    print("Best candidate index:", best_candidate)
    print("Best dot product score:", best_score)

    # Precompute the query-independent bounds (max centroid norms)
    max_norms = precompute_codebook_bounds(codebooks)
    # Simulate candidate codes: each candidate is represented by T indices.
    # Generate a random query vector
    best_candidate, best_score = selective_lookup_dot_with_precomputed_bound(q, candidates_codes, codebooks, max_norms)
    print("Best candidate index:", best_candidate)
    print("Best dot product score:", best_score)


    computed_scores1 = compute_attention_score(q, candidates_codes, codebooks)
    print(computed_scores1)
    computed_scores2 = compute_attention_scores_by_pruning(q, candidates_codes, codebooks, theta=0.01, m=-1.0)

    # Compare the similarity between the two scoring methods
    similarity = np.corrcoef(computed_scores1, computed_scores2)[0,1]
    print("\nCorrelation between regular and pruned attention scores:", similarity)
    
    # Calculate and print the mean absolute difference
    # compute mean relative error between two scores
    # Compute mean relative error between the two scoring methods
    relative_error = np.mean(np.abs(computed_scores1 - computed_scores2) / (np.abs(computed_scores1) + 1e-10))
    print("\nMean relative error between scores:", relative_error)

    
    

    # Train and compute distortions for RQ
    rq = SingleCodebookResidualQuantizer(T=T, k=k)
    print("Fitting Single Codebook Residual Quantizer ...")
    rq.fit(X)
    rq_dist2 = rq.distortion(X, 2.0)
    print("Single Codebok RQ distortion (MSE): {:.6f}".format(rq_dist2))
    #for alpha in np.linspace(1, 3.0, 100):
    #    rq_dist = rq.distortion(X, alpha)
    #    print("alpha: ", alpha)
    #    print("RQ distortion (MSE): {:.6f}".format(rq_dist))

    # Plot a simple bar chart to compare distortions
    #algorithms = ['Product Q.', 'Residual Q.']
    #distortions = [pq_dist, rq_dist]

    #plt.figure(figsize=(6, 4))
    #bars = plt.bar(algorithms, distortions, color=['skyblue', 'salmon'])
    #plt.ylabel('Mean Squared Error')
    #plt.title('PQ vs. RQ Distortion Comparison')
    # Annotate bars with distortion values
    #for bar, mse in zip(bars, distortions):
    #    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{mse:.4f}', 
    #             ha='center', va='bottom', fontsize=10)
    #plt.show()

if __name__ == '__main__':
    main()

