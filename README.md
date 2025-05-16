# rqllm


### Quick PQ/RQ Codebook Training using Random Data and Results

```bash
python ./train_codebooks.py --mode pq --n_subvec 1
python ./train_codebooks.py --mode pq --n_subvec 2
python ./train_codebooks.py --mode rq --L 1
python ./train_codebooks.py --mode rq --L 2
```


| Method | n_subvec/ L  | KL | Att-MSE | Key-MSE |
|--------|-------------|----|---------|---------|
| PQ     | 1           | 2.742e-02 | 3.579e-06 | 2.063e-01 |
| PQ     | 2           | 2.042e-02 | 2.667e-06 | 1.829e-01 |
| RQ     | 1           | 2.746e-02 | 3.583e-06 | 2.064e-01 |
| RQ     | 2           | 1.552e-02 | 2.010e-06 | 1.303e-01 |

###  PQ/RQ Codebook Training using a Larger Random Data and Results
```bash
python ./train_codebooks.py --dataset random64 --mode pq --n_subvec 1
python ./train_codebooks.py --dataset random64 --mode pq --n_subvec 2
python ./train_codebooks.py --dataset random64 --mode pq --n_subvec 3
python ./train_codebooks.py --dataset random64 --mode pq --n_subvec 4
python ./train_codebooks.py --dataset random64 --mode rq --L 1
python ./train_codebooks.py --dataset random64 --mode rq --L 2
python ./train_codebooks.py --dataset random64 --mode rq --L 3
python ./train_codebooks.py --dataset random64 --mode rq --L 4
```
| Method | n_subvec/ L  | KL (before and after KL opt) | Att-MSE | Key-MSE |
|--------|-------------|----|---------|---------|
| PQ     | 1           | 4.245e-02 -> 3.979e-02 |  5.221e-06 | 2.595e-01|
| PQ     | 2           | 3.661e-02 -> 3.484e-02 | 4.595e-06  | 2.284e-01  |
| PQ     | 4           | 2.757e-02 -> 2.632e-02 | 3.497e-06 | 1.787e-01 |
| RQ     | 1           | 4.245e-02 -> 3.979e-02 | 5.214e-06 | 2.595e-01 |
| RQ     | 2           | 3.353e-02 -> 3.029e-02 | 3.995e-06 | 2.120e-01 |
| RQ     | 3           | 2.695e-02 -> 2.397e-02 | 3.165e-06 | 1.785e-01 |
| RQ     | 4           | 2.198e-02 -> 1.945e-02 | 2.469e-06 | 1.533e-01 |
* Att-MSE and Key-MSE are after KL opt

