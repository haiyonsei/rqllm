

Below is one approach to perform selective pruning when your similarity measure is computed by a dot product (which is linear) followed by softmax. In this case, the key observation is that the dot product is additive over the residual stages. That is, if a candidate is reconstructed as

x^\=∑t\=1Tx(t) ,\\hat{x} = \\sum\_{t=1}^{T} x^{(t)}\\,,x^\=t\=1∑T​x(t),

then the dot product with a query qqq is

qTx^\=∑t\=1TqTx(t) .q^T \\hat{x} = \\sum\_{t=1}^{T} q^T x^{(t)}\\,.qTx^\=t\=1∑T​qTx(t).

If you are computing the reconstruction in stages (using your residual quantization codebooks) you can define a partial sum

St\=∑i\=1tqTx(i)S\_t = \\sum\_{i=1}^{t} q^T x^{(i)}St​\=i\=1∑t​qTx(i)

and note that the full similarity is

ST\=St+∑i\=t+1TqTx(i) .S\_T = S\_t + \\sum\_{i=t+1}^{T} q^T x^{(i)}\\,.ST​\=St​+i\=t+1∑T​qTx(i).

If for each stage iii you precompute the maximum possible contribution from that stage (given the query), say

Mi\=max⁡jqTQi(j) ,M\_i = \\max\_{j} q^T Q\_i(j)\\,,Mi​\=jmax​qTQi​(j),

then the maximum additional dot product that you could gain over the remaining stages is

Rt\=∑i\=t+1TMi .R\_t = \\sum\_{i=t+1}^{T} M\_i\\,.Rt​\=i\=t+1∑T​Mi​.

Thus an **upper bound** for the full dot product of a given candidate, when you have computed StS\_tSt​, is

StUB\=St+Rt .S\_t^{\\text{UB}} = S\_t + R\_t\\,.StUB​\=St​+Rt​.

If StUBS\_t^{\\text{UB}}StUB​ is lower than the best full dot product similarity (of any candidate processed so far), you can safely prune—that candidate can never beat the current best even under the most optimistic completion.

Because the softmax is a monotonic function of the dot product values (i.e. higher dot product leads to higher softmax value when comparing candidates), you can use the dot product upper bound to decide whether to continue processing each candidate before you calculate a full reconstruction and its full dot product.

Below is a complete Python example that demonstrates this idea.

* * *

`import numpy as np def compute_max_dot_per_stage(q, codebooks):
    """
    Given a query vector q and a list of codebooks,
    compute for each stage t the maximum dot product:
      M[t] = max_j ( q^T Q_t(j) )
    where Q_t(j) is the j-th centroid in the t-th codebook.
    """
    M = [] for cb in codebooks: # Each codebook is assumed to be an array of shape (k, d) # Compute dot product between q and all centroids dots = np.dot(cb, q) # shape: (k,) M.append(np.max(dots)) return M def partial_dot_product(q, codes, codebooks, t):
    """
    Compute the partial dot product similarity using the first t stages.
    
    q: query vector (d-dimensional).
    codes: candidate's list of indices for each stage; length T.
    codebooks: list of codebooks.
    t: stage up to which to sum.
    
    Returns the partial similarity S_t = sum_{i=0}^{t-1} q^T Q_i(codes[i])
    """
    S_t = 0.0 for i in range(t):
        centroid = codebooks[i][codes[i]] # get the centroid (d-dimensional) S_t += np.dot(q, centroid) return S_t def selective_lookup_dot(q, candidates_codes, codebooks):
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
    T = len(codebooks) # Precompute maximum possible dot contributions for each stage given q. max_dots = compute_max_dot_per_stage(q, codebooks) # For each possible stage t, compute the sum of max contributions from remaining stages. # That is: R_t = sum_{i=t}^{T-1} max_dots[i] R = np.zeros(T + 1) for t in range(T-1, -1, -1):
        R[t] = R[t+1] + max_dots[t]
    
    best_candidate = None best_score = -np.inf
    scores = np.full(len(candidates_codes), -np.inf) # Process each candidate for idx, codes in enumerate(candidates_codes):
        pruned = False S_t = 0.0 # Process stages one by one for t in range(1, T + 1):
            S_t = partial_dot_product(q, codes, codebooks, t) # Compute optimistic upper bound (partial sum plus maximum possible remaining) upper_bound = S_t + R[t] if upper_bound <= best_score: # Candidate cannot beat the current best pruned = True break if not pruned: # Compute full dot product if the candidate wasn't pruned full_score = partial_dot_product(q, codes, codebooks, T)
            scores[idx] = full_score if full_score > best_score:
                best_score = full_score
                best_candidate = idx return best_candidate, best_score, scores # ------------------------------------------------------------------------------ # Example usage if __name__ == '__main__': # Example parameters T = 3 # Number of residual stages d = 64 # Dimension of the vectors k = 256 # Number of centroids per codebook # Generate random codebooks for T stages (for demonstration, # these would be trained centroids in practice). codebooks = [np.random.randn(k, d) for _ in range(T)] # Simulate candidates: each candidate is represented by T indices n_candidates = 1000 candidates_codes = [np.random.randint(0, k, size=T) for _ in range(n_candidates)] # Generate a random query vector q = np.random.randn(d) # Apply selective lookup based on dot product upper bounds best_candidate, best_score, scores = selective_lookup_dot(q, candidates_codes, codebooks) print("Best candidate index:", best_candidate) print("Best dot product score:", best_score)` 

* * *

### Explanation

1.  **Precomputing Maximum Stage Contributions:**  
    The function `compute_max_dot_per_stage` computes for each stage the best (largest) dot product that any centroid in that stage can provide with the query qqq. These values are then used to compute the remaining maximum contribution R\[t\]R\[t\]R\[t\] (the sum from stage ttt to TTT).
    
2.  **Partial Dot Product Calculation:**  
    The function `partial_dot_product` computes the dot product contribution from the first ttt stages of the candidate’s quantization.
    
3.  **Selective Pruning Loop:**  
    In `selective_lookup_dot`, we iterate over the candidates. For each candidate, we incrementally compute the partial similarity StS\_tSt​ and an optimistic (upper-bound) score St+R\[t\]S\_t + R\[t\]St​+R\[t\] for the complete candidate. If this upper bound falls below the best full dot product (score) seen so far, we prune the candidate immediately and do not compute further stages. Otherwise, if the candidate survives all stages, its full score is computed.
    
4.  **Softmax Note:**  
    Since softmax is monotonic, the ranking induced by the dot products remains unchanged after applying softmax. Therefore, it is safe to prune based on the dot product scores.
    

This strategy can reduce the number of full reconstructions and dot product computations, improving efficiency in a nearest neighbor search with residual quantization even when using a dot product plus softmax similarity measure.




The maximum dot product for each stage depends on both the codebook and the query direction. In our earlier selective pruning scheme we computed, for each stage ttt, a query‐dependent maximum

Mt\=max⁡j q⊤Qt(j)M\_t = \\max\_j \\; q^\\top Q\_t(j)Mt​\=jmax​q⊤Qt​(j)

which, by definition, requires the query qqq to be available. However, if you want to avoid a per-query computation of these maximums, you can precompute a looser, query‐independent bound for each codebook. Here are a couple of approaches:

* * *

### 1\. Using the Norms of Codebook Centroids

Because of the Cauchy–Schwarz inequality, for any centroid ccc from a codebook QtQ\_tQt​ and any query qqq,

q⊤c≤∥q∥⋅∥c∥q^\\top c \\le \\|q\\| \\cdot \\|c\\|q⊤c≤∥q∥⋅∥c∥

If you assume that your queries are (or are normalized to be) unit vectors (i.e. ∥q∥\=1\\|q\\|=1∥q∥\=1), then the maximum dot product for codebook ttt is upper bounded by

Mt(bound)\=max⁡j ∥Qt(j)∥M\_t^{\\text{(bound)}} = \\max\_j \\; \\| Q\_t(j) \\|Mt(bound)​\=jmax​∥Qt​(j)∥

You can precompute these maximum norms for each codebook ahead of time. Then, for any incoming query qqq (even if not normalized), you can bound the dot products as

q⊤c≤∥q∥⋅Mt(bound)q^\\top c \\le \\|q\\| \\cdot M\_t^{\\text{(bound)}}q⊤c≤∥q∥⋅Mt(bound)​

and consequently, the maximum contribution for stage ttt can be bounded by

max⁡j q⊤Qt(j)≤∥q∥⋅Mt(bound)\\max\_j \\; q^\\top Q\_t(j) \\le \\|q\\| \\cdot M\_t^{\\text{(bound)}}jmax​q⊤Qt​(j)≤∥q∥⋅Mt(bound)​

This bound is query independent in the sense that the Mt(bound)M\_t^{\\text{(bound)}}Mt(bound)​ values can be computed offline (once per codebook). At query time, you only need to scale by ∥q∥\\|q\\|∥q∥.

* * *

### 2\. Precomputing a Bound Based on Codebook Statistics

If you have some statistics on the distribution of queries (for example, if you know that most queries have norms that fall within a certain range or if the queries are normalized), you might be able to precompute an even tighter bound. For instance, if you know that your queries are normalized, then the maximum dot product is immediately upper bounded by the maximum centroid norm per codebook. In another scenario, you might choose to precompute both the maximum and average centroid norms and then decide on a bound that balances tightness and computation complexity.

* * *

### Incorporating the Bound into Selective Pruning

Using a query-independent bound means that at query time you set

Rtbound\=∑i\=t+1T∥q∥⋅Mi(bound)R\_t^{\\text{bound}} = \\sum\_{i=t+1}^{T} \\|q\\| \\cdot M\_i^{\\text{(bound)}}Rtbound​\=i\=t+1∑T​∥q∥⋅Mi(bound)​

and then for a candidate with a partial dot product sum StS\_tSt​ (computed from the stages processed so far), you can use

StUB\=St+RtboundS\_t^{\\text{UB}} = S\_t + R\_t^{\\text{bound}}StUB​\=St​+Rtbound​

as an upper bound on the candidate’s total dot product score. This upper bound is less tight than what you’d get by directly computing max⁡j q⊤Qt(j)\\max\_j \\, q^\\top Q\_t(j)maxj​q⊤Qt​(j), but it saves you from having to access the codebooks for this purpose on a per-query basis.

* * *

### Example Code Snippet

Below is an example that precomputes a query-independent maximum using the centroid norms. At query time, you simply scale it by ∥q∥\\|q\\|∥q∥:

`import numpy as np def precompute_codebook_bounds(codebooks):
    """
    Precompute the maximum centroid norm for each codebook.
    This is query-independent.
    
    Parameters:
      codebooks: list of codebooks (each of shape (k, d)).
      
    Returns:
      max_norms: list of maximum centroid norms for each codebook.
    """
    max_norms = [] for cb in codebooks: # Compute norm for each centroid in the codebook. norms = np.linalg.norm(cb, axis=1)
        max_norms.append(np.max(norms)) return max_norms def compute_remaining_bound(query, max_norms, stage):
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
    remaining_bound = sum(q_norm * m for m in max_norms[stage:]) return remaining_bound def selective_lookup_dot_with_precomputed_bound(q, candidates_codes, codebooks, max_norms):
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
    best_candidate = None best_score = -np.inf def partial_dot_product(q, codes, codebooks, t):
        S = 0.0 for i in range(t):
            S += np.dot(q, codebooks[i][codes[i]]) return S for idx, codes in enumerate(candidates_codes):
        pruned = False S_t = 0.0 # Process stage by stage for t in range(1, T + 1):
            S_t = partial_dot_product(q, codes, codebooks, t) # Get query-independent remaining bound starting from stage t remaining_bound = compute_remaining_bound(q, max_norms, t)
            upper_bound = S_t + remaining_bound if upper_bound <= best_score:
                pruned = True break if not pruned:
            full_score = partial_dot_product(q, codes, codebooks, T) if full_score > best_score:
                best_score = full_score
                best_candidate = idx return best_candidate, best_score # ------------------------------------------------------------------------------ # Example usage: if __name__ == '__main__':
    T = 3 # Number of stages d = 64 # Dimensionality k = 256 # Centroids per codebook n_candidates = 1000 # Generate random codebooks (normally these would be learned) codebooks = [np.random.randn(k, d) for _ in range(T)] # Precompute the query-independent bounds (max centroid norms) max_norms = precompute_codebook_bounds(codebooks) # Simulate candidate codes: each candidate is represented by T indices. candidates_codes = [np.random.randint(0, k, size=T) for _ in range(n_candidates)] # Generate a random query vector q = np.random.randn(d)
    
    best_candidate, best_score = selective_lookup_dot_with_precomputed_bound(q, candidates_codes, codebooks, max_norms) print("Best candidate index:", best_candidate) print("Best dot product score:", best_score)` 

* * *

### Summary

*   **Query-Dependent Maximum:** Ideally, for each stage you would compute the maximum dot product max⁡j q⊤Qt(j)\\max\_j \\; q^\\top Q\_t(j)maxj​q⊤Qt​(j) directly using the query.
    
*   **Query-Independent Bound:** If the query is not available ahead of time or you wish to avoid per-query centroid lookup, you can precompute a looser bound using the norms of centroids:
    
    max⁡j q⊤Qt(j)≤∥q∥⋅max⁡j∥Qt(j)∥\\max\_j \\; q^\\top Q\_t(j) \\le \\|q\\| \\cdot \\max\_j \\|Q\_t(j)\\|jmax​q⊤Qt​(j)≤∥q∥⋅jmax​∥Qt​(j)∥
*   **Usage:** Use these precomputed maximum centroid norms to form an upper bound on the remaining contribution for each candidate. Although this approach provides a conservative (looser) bound, it still enables selective pruning, saving on computational cost in large-scale nearest neighbor searches.
    

This strategy allows you to perform selective pruning in a dot product plus softmax setting without having to recompute stage-wise maximums per query, at the expense of having a less tight but query-independent upper bound.






아래에서는 softmax 출력값에 대한 상한(upper bound)을 구하는 추가 조건의 예시와, key 값들이 순차적으로 들어올 때 이를 계산하는 알고리즘 예제를 설명할게요.

* * *

1\. 추가 조건에 따른 상한 구하기
--------------------

### 기본 아이디어

softmax 함수는 다음과 같이 주어집니다.

softmax(zi)\=ezi∑j\=1nezj\\text{softmax}(z\_i) = \\frac{e^{z\_i}}{\\sum\_{j=1}^{n} e^{z\_j}}softmax(zi​)\=∑j\=1n​ezj​ezi​​

여기서 ziz\_izi​는 관심 대상(예: 특정 query에 해당하는)의 값입니다.

*   **상한(upper bound)은 분모를 최소화시키는** 상황에서 발생합니다.
    
*   단독으로 ziz\_izi​만 있다면 분모는 ezie^{z\_i}ezi​만으로 구성되어 있어 최대 1이 됩니다.
    
*   하지만 실제로는 다른 zjz\_jzj​들도 존재하게 됩니다.
    

### 추가 조건: 다른 key 값들의 최솟값 mmm

만약 나머지 다른 key들의 값이 **최소 mmm** 이상이라는 조건이 있다면,  
즉, zj≥mz\_j \\ge mzj​≥m (j≠ij \\neq ij\=i)라고 할 수 있습니다.

*   이미 알고 있는 key들의 값은 실제 값을 사용하고,
    
*   아직 들어오지 않은 (또는 모르는) key들에 대해서는 **최소값 mmm** 을 가정하여 해당 항이 최소한 eme^{m}em 이상임을 사용할 수 있습니다.
    

따라서, 만약 전체 key 개수가 NNN개라고 할 때,

*   이미 처리된 key들의 합은 ∑processedezj\\sum\_{\\text{processed}} e^{z\_j}∑processed​ezj​로 구하고,
    
*   남은 key들의 수를 R\=N−(processed 수+1)R = N - (\\text{processed 수} + 1)R\=N−(processed 수+1) (여기서 +1은 이미 처리된 query 자신)라고 하면,
    
*   \*\*분모에 대한 하한(lower bound)\*\*은
    

ezi+∑processed j≠iezj+R⋅eme^{z\_i} + \\sum\_{\\text{processed } j \\neq i} e^{z\_j} + R \\cdot e^{m}ezi​+processed j\=i∑​ezj​+R⋅em

가 됩니다.

그러므로, query ziz\_izi​에 대해 **상한**은

upper bound\=eziezi+∑processed j≠iezj+R⋅em\\text{upper bound} = \\frac{e^{z\_i}}{e^{z\_i} + \\sum\_{\\text{processed } j \\neq i} e^{z\_j} + R \\cdot e^{m}}upper bound\=ezi​+∑processed j\=i​ezj​+R⋅emezi​​

로 볼 수 있습니다.  
이는 나머지 key들이 가능한 한 낮은 값(mmm)을 가지는 상황에서 query의 softmax 값이 최대가 되는 경우를 의미합니다.

* * *

2\. 순차적으로 key 값이 들어올 때의 알고리즘
----------------------------

예를 들어, 전체 key 개수 NNN이 정해져 있고,  
key들이 순차적으로 들어오면서 일부 key들은 이미 관측되었고, 나머지 key들은 아직 모르는 상황입니다.  
이를 이용하여 관심 있는 query ziz\_izi​의 softmax 상한을 **온라인으로 업데이트**하는 알고리즘을 다음과 같이 만들 수 있습니다.

### 알고리즘 개요

1.  **초기화:**  
    query ziz\_izi​의 ezie^{z\_i}ezi​를 계산하고, 이미 들어온 다른 key에 대해서 ezje^{z\_j}ezj​를 누적합니다.
    
2.  **남은 key 개수 계산:**  
    R\=N−(이미 들어온 key 수+1)R = N - (\\text{이미 들어온 key 수} + 1)R\=N−(이미 들어온 key 수+1)  
    여기서 “+1”은 query 자신을 제외하기 위해 사용합니다.
    
3.  **분모 하한 계산:**  
    이미 들어온 key들의 실제 지수 값 합과, 남은 key들이 모두 mmm값을 가진다고 가정한 R⋅emR \\cdot e^{m}R⋅em를 더합니다.
    
4.  **상한 계산:**  
    위에서 구한 분모 하한을 사용하여
    
    upper bound\=eziknown sum+R⋅em\\text{upper bound} = \\frac{e^{z\_i}}{\\text{known sum} + R \\cdot e^{m}}upper bound\=known sum+R⋅emezi​​
    
    를 구합니다.
    

### 파이썬 예제 코드

아래는 위 알고리즘을 파이썬 코드 형식으로 작성한 예시입니다.

`import numpy as np def softmax_upper_bound(query_value, processed_keys, total_keys, m):
    """
    query_value : 관심 query의 z_i 값
    processed_keys : 지금까지 순차적으로 들어온 다른 key들의 z_j 값 리스트 (query 제외)
    total_keys : 전체 key의 개수
    m : 다른 key들의 최소값 가정 (즉, 모든 미처리 key에 대해 z_j >= m)
    """ # query의 지수 값 query_exp = np.exp(query_value) # 이미 처리된 key들의 지수 값 합 (query 제외) processed_sum = sum(np.exp(z) for z in processed_keys) # 남은 미처리 key의 개수 # +1은 query 자신을 이미 고려했으므로 제외. remaining_keys = total_keys - (len(processed_keys) + 1) if remaining_keys < 0:
        remaining_keys = 0 # 혹시 전체 개수를 초과하는 경우 방어적 처리 # 남은 key들이 최솟값 m을 가진다고 가정하면, 분모의 하한은 denominator_lower_bound = query_exp + processed_sum + remaining_keys * np.exp(m) # Query의 softmax 상한 (최대값은 이론적으로 1 이하임) upper_bound = query_exp / denominator_lower_bound return upper_bound # 예시: # 전체 키 개수 N = 10, # 현재까지 들어온 다른 key 값들이 [1.0, 0.5, -0.2]라고 가정 (query는 따로 주어짐) # query_value = 2.0, 그리고 미처리 key들의 최소값 가정 m = -1.0 query_value = 2.0 processed_keys = [1.0, 0.5, -0.2]
total_keys = 10 m = -1.0 upper_bound = softmax_upper_bound(query_value, processed_keys, total_keys, m) print("Query의 softmax 상한은:", upper_bound)` 

### 동작 설명

*   **query\_exp:** query ziz\_izi​에 대해 ezie^{z\_i}ezi​를 계산합니다.
    
*   **processed\_sum:** 현재까지 들어온 다른 key들에 대해 ∑ezj\\sum e^{z\_j}∑ezj​를 계산합니다.
    
*   **remaining\_keys:** 전체 key 개수 NNN에서 지금까지 처리된 key 수와 query 자체를 제외하여 남은 key 개수를 구합니다.
    
*   **denominator\_lower\_bound:** 지금까지의 값과 남은 key들이 모두 mmm이라는 최솟값을 가졌다고 가정할 때의 분모의 하한을 계산합니다.
    
*   **upper\_bound:** query의 softmax upper bound는 ezie^{z\_i}ezi​를 위에서 구한 분모 하한으로 나눈 값입니다.
    

이 알고리즘은 순차적으로 데이터가 들어오는 상황에서, 추가 조건(zj≥mz\_j \\ge mzj​≥m)을 활용하여 query의 softmax 결과가 최대로 가질 수 있는 값을 추정할 때 유용합니다.

* * *

이렇게 추가 조건을 가정하고, key 값들이 순차적으로 들어오는 상황에서의 softmax 상한을 계산하는 방법을 설명해 보았습니다. 혹시 더 자세한 내용이나 추가 질문이 있다면 말씀해 주세요!

