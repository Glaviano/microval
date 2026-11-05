from .samplers.wes import WES, pre_compute_wes
from .samplers.biwes import BiWES, pre_compute_biwes 
from .samplers.west import WEST, pre_compute_west
from .samplers.biwest import BiWEST, pre_compute_biwest
from numba import njit, prange
import numba
import numpy as np


@njit(parallel=True)
def get_pval_WEST(A, n_iterations, n_run, n_threads):
    """
    Compute empirical p-values for an undirected weighted graph using the WEST algorithm in parallel.

    Args:
        A (np.ndarray): 2D square adjacency matrix of the undirected network.
        n_iterations (int): Number of randomization moves per simulation.
        n_run (int): Total number of independent simulations.
        n_threads (int): Number of parallel threads to use.

    Returns:
        p_values (np.ndarray): 2D matrix of empirical p-values for each edge.
    """
    n = A.shape[0]
    A_obs = A.copy()
    n_link = int(np.sum(A_obs != 0) / 2)  # Initial number of undirected links
    edgelist_numpy, n_link = pre_compute_west(A_obs)
    # Distribute simulations across threads
    sims_per_thread = n_run // n_threads
    remainder = n_run % n_threads
    
    thread_local_counts = np.zeros((n_threads, n, n), dtype=np.int32)
    for tid in prange(n_threads):
        n_sims = sims_per_thread + (1 if tid < remainder else 0)
        
        # Each thread starts with a fresh copy of A
        A_current = A.copy()
        edgelist_current, n_link = pre_compute_west(A_current)
        n_link_current = n_link
        # Simulation chain: output(t) -> input(t+1)
        for sim in range(n_sims):
            # Run simulation (modifies A_current and data structures in-place)
            A_current, edgelist_current, n_link_current = WEST(A_current, edgelist_current, n_link_current, n_iterations)
            
            for i in range(n_link):
                a, b = edgelist_numpy[i]
                if A_current[a, b] >= A_obs[a, b]:
                    thread_local_counts[tid, a, b] += 1
                    thread_local_counts[tid, b, a] += 1
        
    # Combine results from all threads
    total_counts = np.sum(thread_local_counts, axis=0)
    for i in range(n):
        for j in range(i+1, n, 1):
            if A_obs[i, j] == 0:
                total_counts[i, j] = n_run
                total_counts[j, i] = n_run
    return total_counts / float(n_run)


@njit(parallel=True)
def get_pval_BiWEST(A, n_iterations, n_run, n_threads):
    """
    Compute empirical p-values for a bipartite weighted graph using the BiWEST algorithm in parallel.

    Args:
        A (np.ndarray): 2D bi-adjacency matrix of shape (na, nb).
        n_iterations (int): Number of randomization moves per simulation.
        n_run (int): Total number of independent simulations.
        n_threads (int): Number of parallel threads to use.

    Returns:
        p_values (np.ndarray): 2D matrix of empirical p-values for each edge.
    """
    na = A.shape[0]
    nb = A.shape[1]
    A_obs = A.copy()
    n_link = int(np.sum(A_obs != 0))  # Initial number of bipartite links
    edgelist_numpy, n_edges = pre_compute_biwest(A_obs)
    # Distribute simulations across threads
    sims_per_thread = n_run // n_threads
    remainder = n_run % n_threads

    thread_local_counts = np.zeros((n_threads, na, nb), dtype=np.int32)
    for tid in prange(n_threads):
        n_sims = sims_per_thread + (1 if tid < remainder else 0)
        
        # Each thread starts with a fresh copy of A
        A_current = A.copy()
        edgelist_current, n_link_current = pre_compute_biwest(A_current)
        # Simulation chain: output(t) -> input(t+1)
        for sim in range(n_sims):
            # Run simulation (modifies A_current and data structures in-place)
            A_current, edgelist_current, n_link_current = BiWEST(A_current, edgelist_current, n_link_current, n_iterations)
            
            for i in range(n_link):
                a, b = edgelist_numpy[i]
                if A_current[a, b] >= A_obs[a, b]:
                    thread_local_counts[tid, a, b] += 1
        
    # Combine results from all threads
    total_counts = np.sum(thread_local_counts, axis=0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A_obs[i, j] == 0:
                total_counts[i, j] = n_run
    return total_counts / float(n_run)


@njit(parallel=True)
def get_pval_BiWES(A, n_iterations, n_run, n_threads):
    """
    Compute empirical p-values for a bipartite weighted graph using the BiWES algorithm in parallel.

    Args:
        A (np.ndarray): 2D bi-adjacency matrix of shape (na, nb).
        n_iterations (int): Number of randomization moves per simulation.
        n_run (int): Total number of independent simulations.
        n_threads (int): Number of parallel threads to use.

    Returns:
        p_values (np.ndarray): 2D matrix of empirical p-values for each edge.
    """
    na, nb = A.shape
    A_obs = A.copy()
    
    # Distribute simulations across threads
    sims_per_thread = n_run // n_threads
    remainder = n_run % n_threads
    
    thread_local_counts = np.zeros((n_threads, na, nb), dtype=np.int32)

    for tid in prange(n_threads):
        n_sims = sims_per_thread + (1 if tid < remainder else 0)
        
        # Each thread starts with its own copy of A and pre-computed structures
        A_current = A.copy()
        A2, weight_dict, weight_dict2, adj_list, adj_list2, unique_weights, unique_weights2 = pre_compute_biwes(A_current)
        
        # Simulation chain: output(t) -> input(t+1)
        for sim in range(n_sims):
            A_current = BiWES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations)
            
            # Count events
            for i in range(na):
                for j in range(nb):
                    if A_current[i, j] >= A_obs[i, j]:
                        thread_local_counts[tid, i, j] += 1
        
    # Combine results from all threads
    total_counts = np.sum(thread_local_counts, axis=0)
    return total_counts / float(n_run)


@njit(parallel=True)
def get_pval_WES(A, n_iterations, n_run, n_threads):
    """
    Compute empirical p-values for an undirected weighted graph using the WES algorithm in parallel.

    Args:
        A (np.ndarray): 2D square adjacency matrix of the undirected network.
        n_iterations (int): Number of randomization moves per simulation.
        n_run (int): Total number of independent simulations.
        n_threads (int): Number of parallel threads to use.

    Returns:
        p_values (np.ndarray): 2D matrix of empirical p-values for each edge.
    """
    na, nb = A.shape
    A_obs = A.copy()
    
    # Divide the total number of simulations among the available threads
    sims_per_thread = n_run // n_threads
    remainder = n_run % n_threads
    
    thread_local_counts = np.zeros((n_threads, na, nb), dtype=np.int32)

    for tid in prange(n_threads):
        n_sims = sims_per_thread + (1 if tid < remainder else 0)
        
        # Each thread works on its own copy to avoid race conditions
        A_current = A.copy()
        A2, weight_dict, weight_dict2, adj_list, adj_list2, unique_weights, unique_weights2 = pre_compute_wes(A_current)
        
        for sim in range(n_sims):
            A_current = WES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations)
            
            # Count how many times each edge has weight greater than or equal to observed
            for i in range(na):
                for j in range(i+1, na, 1):
                    if A_current[i, j] >= A_obs[i, j]:
                        thread_local_counts[tid, i, j] += 1
                        thread_local_counts[tid, j, i] += 1                
        
    # Merge counts from all threads
    total_counts = np.sum(thread_local_counts, axis=0)

    return total_counts / float(n_run)


def compute_pval(A, method, n_iterations, n_run, n_threads=None):
    """
    Compute empirical p-values for edge weights by randomizing the graph using the specified method.

    Args:
        A (np.ndarray): Adjacency matrix or bi-adjacency matrix of the network.
        method (str): Sampling method to use ('BiWES', 'BiWEST', 'WES', 'WEST').
        n_iterations (int): Number of swap/randomization moves per simulation run.
        n_run (int): Number of randomized graph samples.
        n_threads (int, optional): Number of parallel threads to use. If None, uses all available threads.

    Returns:
        P (np.ndarray): Matrix of empirical p-values.
    """
    if n_threads is not None:
        numba.set_num_threads(int(n_threads))
    threads_to_use = numba.get_num_threads()
    if method == "BiWES":
        P = get_pval_BiWES(A, n_iterations, n_run, threads_to_use)
    elif method == "BiWEST":
        P = get_pval_BiWEST(A, n_iterations, n_run, threads_to_use)
    elif method == "WES":
        P = get_pval_WES(A, n_iterations, n_run, threads_to_use)
    elif method == "WEST":
        P = get_pval_WEST(A, n_iterations, n_run, threads_to_use)
    else:
        raise ValueError("Invalid method. Choose from 'BiWES', 'BiWEST', 'WES', 'WEST'.")
    return P


def _fdr_bh(pvals, alpha):
    """
    Benjamini-Hochberg False Discovery Rate (FDR) correction in pure NumPy.

    Args:
        pvals (np.ndarray): Array of uncorrected p-values.
        alpha (float): Significance threshold level.

    Returns:
        reject (np.ndarray): Boolean array indicating whether each null hypothesis is rejected.
    """
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_pvals = pvals[order]
    threshold = (np.arange(1, n + 1) / n) * alpha
    significant = sorted_pvals <= threshold
    reject = np.zeros(n, dtype=bool)
    if np.any(significant):
        max_k = np.max(np.where(significant)[0])
        reject[order[:max_k + 1]] = True
    return reject


def validate_network(A, P, alpha=0.05, is_bipartite=False, verbose=False):
    """
    Validate edges of a weighted network using Bonferroni and FDR (Benjamini-Hochberg) corrections.

    Args:
        A (np.ndarray): Original adjacency matrix (square/symmetric) or bi-adjacency matrix.
        P (np.ndarray): Empirical p-value matrix (same shape as A).
        alpha (float, optional): Significance level (default: 0.05).
        is_bipartite (bool, optional): Whether the network is bipartite (default: False).
        verbose (bool, optional): If True, prints validation summary statistics.

    Returns:
        A_bonf (np.ndarray): Validated network filtered with Bonferroni correction.
        A_fdr (np.ndarray): Validated network filtered with FDR (BH) correction.
    """
    if is_bipartite:
        # For bipartite graphs: test all existing edges (A != 0)
        mask = A != 0
        pvalues = P[mask]
        m = len(pvalues)
        
        # 1. Bonferroni correction
        reject_bonf = pvalues <= (alpha / m)
        A_bonf = np.zeros_like(A)
        A_bonf[mask] = np.where(reject_bonf, A[mask], 0)
        
        # 2. FDR Benjamini-Hochberg correction
        reject_fdr = _fdr_bh(pvalues, alpha)
        A_fdr = np.zeros_like(A)
        A_fdr[mask] = np.where(reject_fdr, A[mask], 0)
        if verbose:
            print(f"Tested links: {m}")
            print(f"Validated links (Bonferroni): {np.count_nonzero(A_bonf)}")
            print(f"Validated links (FDR): {np.count_nonzero(A_fdr)}")
            
    else:
        # For undirected unipartite graphs: test only upper triangular edges
        rows, cols = np.triu_indices_from(A, k=1)
        existing_edges = A[rows, cols] != 0
        
        rows = rows[existing_edges]
        cols = cols[existing_edges]
        pvalues = P[rows, cols]
        m = len(pvalues)
        
        # 1. Bonferroni correction
        reject_bonf = pvalues <= (alpha / m)
        A_bonf = np.zeros_like(A)
        valid_r_bonf = rows[reject_bonf]
        valid_c_bonf = cols[reject_bonf]
        A_bonf[valid_r_bonf, valid_c_bonf] = A[valid_r_bonf, valid_c_bonf]
        A_bonf[valid_c_bonf, valid_r_bonf] = A[valid_r_bonf, valid_c_bonf]  # Symmetric
        
        # 2. FDR Benjamini-Hochberg correction
        reject_fdr = _fdr_bh(pvalues, alpha)
        A_fdr = np.zeros_like(A)
        valid_r_fdr = rows[reject_fdr]
        valid_c_fdr = cols[reject_fdr]
        A_fdr[valid_r_fdr, valid_c_fdr] = A[valid_r_fdr, valid_c_fdr]
        A_fdr[valid_c_fdr, valid_r_fdr] = A[valid_r_fdr, valid_c_fdr]  # Symmetric
        if verbose:
            print(f"Tested links: {m}")
            print(f"Validated links (Bonferroni): {np.count_nonzero(A_bonf) // 2}")
            print(f"Validated links (FDR): {np.count_nonzero(A_fdr) // 2}")
    return A_bonf, A_fdr
          
   
