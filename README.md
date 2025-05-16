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

