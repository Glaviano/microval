import numpy as np
from numba import njit


@njit
def WEST(A, edgelist_array, n_link, n_step):
    """
    Randomize a weighted undirected graph using the WEST algorithm, preserving the strength sequence.

    Args:
        A (np.ndarray): 2D square adjacency matrix of the undirected graph (modified in-place).
        edgelist_array (np.ndarray): Pre-allocated 2D array of shape (max_edges, 2) storing the active edges.
        n_link (int): Current number of active edges in edgelist_array.
        n_step (int): Number of randomization steps (iterations) to perform.

    Returns:
        A (np.ndarray): The randomized adjacency matrix.
        edgelist_array (np.ndarray): The updated edge list array.
        n_link (int): The updated number of active edges in the edge list.
    """
    for _ in range(n_step):
        # 1. Select two distinct edges
        idx1 = np.random.randint(n_link)
        idx2 = np.random.randint(n_link)
        while idx2 == idx1:
            idx2 = np.random.randint(n_link)
            
        a, b = edgelist_array[idx1]
        c, d = edgelist_array[idx2]
        
        w_ab = A[a, b]
        w_cd = A[c, d]
        
        if a == c or a == d or b == c or b == d:
            continue
            
        minimum = w_ab if w_ab < w_cd else w_cd
        
        # 2. Select a random weight increment
        w = np.random.randint(1, minimum + 1)
        
        new_n_link = n_link
        new_w_ab = w_ab - w
        new_w_cd = w_cd - w
        
        if new_w_ab == 0:
            new_n_link -= 1
        if new_w_cd == 0:
            new_n_link -= 1
        
        # 3. Random pairing: choose between (c, d) or (d, c)
        if np.random.randint(2) == 0:
            k, l = c, d
        else:
            k, l = d, c

        # 4. Check target edge weights
        old_w_ak = A[a, k]
        old_w_bl = A[b, l]
        w_ak = old_w_ak + w
        w_bl = old_w_bl + w
        
        minimum2 = w_ak if w_ak < w_bl else w_bl
        
        if old_w_ak == 0:
            new_n_link += 1
        if old_w_bl == 0:
            new_n_link += 1
            
        # 5. Acceptance probability (Metropolis-Hastings ratio)
        num = float(n_link * (n_link - 1) * minimum)
        den = float(new_n_link * (new_n_link - 1) * minimum2)
        P = num / den
        
        if np.random.rand() < P:
            # Apply changes to the adjacency matrix
            A[a, b] = new_w_ab
            A[c, d] = new_w_cd
            A[a, k] = w_ak
            A[b, l] = w_bl
            
            A[b, a] = new_w_ab
            A[d, c] = new_w_cd
            A[k, a] = w_ak
            A[l, b] = w_bl
            
            # 6. Remove edges with weight 0 using swap-and-pop in O(1)
            # Remove higher index first to preserve the position of the lower index
            if new_w_ab == 0 and new_w_cd == 0:
                idx_max = idx1 if idx1 > idx2 else idx2
                idx_min = idx1 if idx1 < idx2 else idx2
                
                # Pop higher index
                edgelist_array[idx_max] = edgelist_array[n_link - 1]
                n_link -= 1
                # Pop lower index
                edgelist_array[idx_min] = edgelist_array[n_link - 1]
                n_link -= 1
            elif new_w_ab == 0:
                edgelist_array[idx1] = edgelist_array[n_link - 1]
                n_link -= 1
            elif new_w_cd == 0:
                edgelist_array[idx2] = edgelist_array[n_link - 1]
                n_link -= 1
                
            # 7. Add newly created edges
            if old_w_ak == 0:
                edgelist_array[n_link, 0] = a
                edgelist_array[n_link, 1] = k
                n_link += 1
            if old_w_bl == 0:
                edgelist_array[n_link, 0] = b
                edgelist_array[n_link, 1] = l
                n_link += 1
                
    return A, edgelist_array, n_link


@njit
def pre_compute_west(A):
    """
    Pre-compute the initial edge list data structures for the WEST algorithm from an adjacency matrix.

    Args:
        A (np.ndarray): 2D square adjacency matrix of the undirected graph.

    Returns:
        edgelist (np.ndarray): Pre-allocated 2D array of shape (max_edges, 2) containing active edges.
        count (int): Initial number of active edges in the graph.
    """
    n = A.shape[0]
    E_max = n * (n - 1) // 2  # Maximum number of edges in an undirected graph
    edgelist = np.empty((E_max, 2), dtype=np.int64)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                edgelist[count] = (i, j)
                count += 1
    
    return edgelist, count




