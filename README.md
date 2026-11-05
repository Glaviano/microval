# microval

A Python package for **randomization and statistical validation of weighted unipartite and bipartite networks**.

`microval` implements the **WEST**, **BiWEST**, **WES**, and **BiWES** null models using Numba. These algorithms generate a sample of random matrices, each satisfying a specified set of constraints. The sample is then used to estimate the matrix of $p$-values for statistical validation.

---

## Overview of Null Models & Samplers

Each sampler in `microval` is designed for a specific class of networks and preserves an exact set of network properties during randomization:

| Method | Network Type | Invariants Preserved | Matrix Representation |
| :--- | :--- | :--- | :--- |
| **WEST** | Undirected Weighted | **Strength sequence** of all nodes | Symmetric square matrix ($n \times n$) |
| **BiWEST** | Undirected Bipartite Weighted | **Strength sequence** of both node sets | Rectangular bi-adjacency matrix ($n_a \times n_b$) |
| **WES** | Undirected Weighted | **Degree sequence** and **Strenght sequence** | Symmetric square matrix ($n \times n$) |
| **BiWES** | Undirected Bipartite Weighted | **Degree sequence** and **Strenght sequence** of both node sets| Rectangular bi-adjacency matrix ($n_a \times n_b$) |

*(The netwrok must have only integer weights)*


## Installation

The module can be installed as a standard package:

```bash
pip install microval
```

### Dependencies
- `numpy >= 1.21.0`
- `numba >= 0.56.0`


---

## Workflow 1: Using Samplers Directly

You can directly generate randomized networks for each of the 4 null models.

### 1. WEST (Weighted Edge Swap with Threshold)
- **Applicability:** Undirected weighted networks.
- **Preserves:** The exact **strength sequence** (sum of weights per node). Edge weights and topology can change dynamically.
- **Workflow:**
  1. `pre_compute_west(A)`
  2. `WEST(A, edgelist, n_link, n_step)`

```python
import numpy as np
from microval.samplers import WEST, pre_compute_west

# Symmetric weighted adjacency matrix
A = np.array([
    [0, 5, 2, 0],
    [5, 0, 0, 4],
    [2, 0, 0, 3],
    [0, 4, 3, 0]
], dtype=np.int64)

# 1. Pre-compute edge list structures
edgelist, n_link = pre_compute_west(A)

# 2. Perform randomization steps (in-place on a copy)
A_rand, edgelist, n_link = WEST(A.copy(), edgelist, n_link, n_step=1000)

print("WEST randomized matrix:")
print(A_rand)
```

---

### 2. BiWEST (Bipartite Weighted Edge Swap with Threshold)
- **Applicability:** Bipartite weighted networks.
- **Preserves:** The exact **row strength sequence** and **column strength sequence** across both partitions.
- **Workflow:**
  1. `pre_compute_biwest(A)`
  2. `BiWEST(A, edgelist, n_link, n_step)`

```python
import numpy as np
from microval.samplers import BiWEST, pre_compute_biwest

# Bi-adjacency matrix (na x nb)
B = np.array([
    [10,  0,  3],
    [ 0,  7,  2],
    [ 4,  5,  0]
], dtype=np.int64)

# 1. Pre-compute bipartite edge list structures
edgelist, n_link = pre_compute_biwest(B)

# 2. Randomize the bipartite network
B_rand, edgelist, n_link = BiWEST(B.copy(), edgelist, n_link, n_step=1000)

print("BiWEST randomized matrix:")
print(B_rand)
```

---

### 3. WES (Weighted Edge Swap)
- **Applicability:** Undirected weighted networks.
- **Preserves:** 
  - The **degree sequence** (number of connections per node).
  - The **weight sequence** (the exact multiset of link weights present in the network).
  - The **strength sequence** (as a consequence of the two conservation laws above).
- **Workflow:**
  1. `pre_compute_wes(A)`
  2. `WES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations)`

```python
import numpy as np
from microval.samplers import WES, pre_compute_wes

A = np.array([
    [0, 5, 2, 0],
    [5, 0, 0, 4],
    [2, 0, 0, 3],
    [0, 4, 3, 0]
], dtype=np.int64)

# 1. Pre-compute Numba typed structures (adjacency lists, weight maps)
A2, weight_dict, weight_dict2, adj_list, adj_list2, unique_weights, unique_weights2 = pre_compute_wes(A)

# 2. Randomize the network
A_rand = WES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations=1000)

print("WES randomized matrix:")
print(A_rand)
```

---

### 4. BiWES (Bipartite Weighted Edge Swap)
- **Applicability:** Bipartite weighted networks.
- **Preserves:**
  - The **degree sequence** of both layers.
  - The exact **weight sequence** (all edge weights are conserved and rearranged).
  - The **strength sequence** of both layers.
- **Workflow:**
  1. `pre_compute_biwes(A)`
  2. `BiWES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations)`

```python
import numpy as np
from microval.samplers import BiWES, pre_compute_biwes

B = np.array([
    [10,  0,  3],
    [ 0,  7,  2],
    [ 4,  5,  0]
], dtype=np.int64)

# 1. Pre-compute bipartite Numba structures
A2, weight_dict, weight_dict2, adj_list, adj_list2, unique_weights, unique_weights2 = pre_compute_biwes(B)

# 2. Randomize the bipartite network
B_rand = BiWES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations=1000)

print("BiWES randomized matrix:")
print(B_rand)
```

---

## Workflow 2: Statistical Network Validation

For edge validation, `microval` automates parallel graph generation, Monte Carlo $p$-value estimation, and multiple hypothesis testing corrections.

### Complete Example

```python
import numpy as np
from microval import compute_pval, validate_network

# 1. Undirected network validation
A = np.array([
    [0, 12, 1, 0, 0],
    [12, 0, 8, 2, 0],
    [1, 8, 0, 15, 0],
    [0, 2, 15, 0, 7],
    [0, 0, 0, 7, 0]
], dtype=np.int64)

# Compute p-values with parallel multithreading (4 threads)
# Supported methods: 'WEST', 'WES', 'BiWEST', 'BiWES'
P_matrix = compute_pval(A, method="WEST", n_iterations=500, n_run=1000, n_threads=4)

# Filter edges using Bonferroni and Benjamini-Hochberg FDR (alpha = 0.05)
A_bonferroni, A_fdr = validate_network(A, P_matrix, alpha=0.05, is_bipartite=False, verbose=True)

print("\nValidated Network (FDR):")
print(A_fdr)
```

---

## API Summary

### Validation Functions

#### `compute_pval(A, method, n_iterations, n_run, n_threads=None)`
- **`A`** (`np.ndarray`): Adjacency matrix (square for unipartite) or bi-adjacency matrix (rectangular for bipartite).
- **`method`** (`str`): One of `'WEST'`, `'BiWEST'`, `'WES'`, `'BiWES'`.
- **`n_iterations`** (`int`): Number of swap/randomization moves per simulation run.
- **`n_run`** (`int`): Number of independent randomized graph samples generated.
- **`n_threads`** (`int`, optional): Number of parallel CPU threads to use (defaults to all available cores).
- **Returns**: `P` (`np.ndarray`): Matrix of empirical $p$-values with the same dimensions as `A`.

#### `validate_network(A, P, alpha=0.05, is_bipartite=False, verbose=False)`
- **`A`** (`np.ndarray`): Original weighted network matrix.
- **`P`** (`np.ndarray`): Matrix of empirical $p$-values.
- **`alpha`** (`float`, optional): Significance threshold level (default `0.05`).
- **`is_bipartite`** (`bool`, optional): Set to `True` for bipartite networks (default `False`).
- **`verbose`** (`bool`, optional): Prints summary of tested and validated links (default `False`).
- **Returns**: `(A_bonf, A_fdr)`: Tuple of validated networks filtered respectively with Bonferroni and Benjamini-Hochberg FDR corrections.

---

### Sampler Functions

| Method | Pre-compute Function | Main Sampler Function |
| :--- | :--- | :--- |
| **WEST** | `pre_compute_west(A)` | `WEST(A, edgelist, n_link, n_step)` |
| **BiWEST** | `pre_compute_biwest(A)` | `BiWEST(A, edgelist, n_link, n_step)` |
| **WES** | `pre_compute_wes(A)` | `WES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations)` |
| **BiWES** | `pre_compute_biwes(A)` | `BiWES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations)` |

---

## License

This project is licensed under the MIT License.
