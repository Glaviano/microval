import numba.types as nt
import numpy as np
from numba.typed import Dict, List
from numba import njit
USE_NJIT = True

#GLOBAL CONFIGURATION FOR NUMBA
def conditional_njit(func):
    if USE_NJIT:
        return njit(cache=True)(func)
    else:
        return func

# ==========================================
# DEFINING NUMBA TYPES THAT WILL BE USED THROUGHOUT THE CODE
# ==========================================
node_type = nt.int64
neighbors_type = nt.DictType(node_type, node_type)
edge_type = nt.UniTuple(node_type, 2)
edge_list_dict_type = nt.DictType(edge_type, node_type)
list_node_type = nt.ListType(node_type)
edge_list_ty = nt.ListType(edge_type)
rect_type = nt.UniTuple(node_type, 4)
list_dict_rect = nt.DictType(rect_type, node_type)

@conditional_njit
def from_adj_matrix_to_adj_list(A):
    """
    Create two adjacency lists representation for an undirected graph given its adjacency matrix.

    Args:
        A (np.ndarray): 2D square adjacency matrix of the undirected graph.

    Returns:
        adj_list (Dict[int, Dict[int, int]]): Dictionary mapping each node to a dictionary of its neighbors and their corresponding indices in adj_list2.
        adj_list2 (Dict[int, List[int]]): Dictionary mapping each node to a list of its neighbors.
    """
    n = A.shape[0]
    adj_list = Dict.empty(
        key_type=node_type,
        value_type=neighbors_type
    )
    
    adj_list2 = Dict.empty(
        key_type=node_type,
        value_type=list_node_type
    )

    for i in range(n):
        neighbors = Dict.empty(
            key_type=node_type,
            value_type=node_type
        )
        cont = 0
        vicini = List.empty_list(node_type)
        for j in range(n):
            if A[i, j] != 0:
                neighbors[j] = cont
                vicini.append(j)
                cont += 1
        adj_list[i] = neighbors
        adj_list2[i] = vicini
    

    return adj_list, adj_list2



@conditional_njit
def get_weight_dict(A):
    """
    Construct weight dictionaries and unique weight trackers from an undirected adjacency matrix.

    Args:
        A (np.ndarray): 2D square adjacency matrix of the undirected graph.

    Returns:
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Mapping of weight to dict of (edge -> index in weight_dict2).
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Mapping of weight to list of edges with that weight.
        unique_weights (Dict[int, int]): Mapping of unique weight to its index in unique_weights2.
        unique_weights2 (List[int]): List of unique edge weights present in the graph.
    """
    n = A.shape[0]
    weight_dict = Dict.empty(
        key_type=node_type,
        value_type= edge_list_dict_type
    )
    weight_dict2 = Dict.empty(
        key_type=node_type,
        value_type= edge_list_ty
    )
    unique_weights = Dict.empty(
        key_type=node_type,
        value_type=node_type
    )
    unique_weights2 = List.empty_list(node_type)
    for i in range(n):
        for j in range(i+1, n, 1):
            if A[i, j] != 0:
                weight = A[i, j]
                edge = (i, j)
                if weight not in unique_weights:
                    unique_weights2.append(weight)
                    unique_weights[weight] = len(unique_weights2) - 1
                if weight not in weight_dict:
                    weight_dict[weight] = Dict.empty(
                        key_type=edge_type,
                        value_type=node_type
                    )
                if weight not in weight_dict2:
                    weight_dict2[weight] = List.empty_list(edge_type)
                weight_dict2[weight].append(edge)

                weight_dict[weight][edge] = len(weight_dict2[weight]) - 1

    return weight_dict, weight_dict2, unique_weights, unique_weights2


@conditional_njit
def update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, old_links, new_links):
    """
    Update weight dictionaries and unique weights when edges change weights or are replaced.

    Args:
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Mapping of weight to dict of (edge -> index).
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Mapping of weight to list of edges.
        unique_weights (Dict[int, int]): Mapping of unique weight to its index.
        unique_weights2 (List[int]): List of unique weights.
        old_weight (np.ndarray): Array of old edge weights to remove or decrement.
        new_weight (np.ndarray): Array of new edge weights to insert.
        old_links (List[Tuple[int, int]]): Edges corresponding to old weights.
        new_links (List[Tuple[int, int]]): Edges corresponding to new weights.

    Returns:
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Updated weight dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Updated edge list dictionary by weight.
        unique_weights (Dict[int, int]): Updated unique weights mapping.
        unique_weights2 (List[int]): Updated list of unique weights.
    """
    # Remove old weights
    # Loop over old weights to remove them from the weight dictionaries and unique weights if necessary
    for i in range(len(old_weight)):
        weight = old_weight[i]
        edge = old_links[i]
        if weight in weight_dict:
            index = weight_dict[weight][edge]
            del weight_dict[weight][edge]
            last_edge = weight_dict2[weight][-1]
            if edge != last_edge:
                weight_dict2[weight][index] = last_edge
                weight_dict[weight][last_edge] = index
            weight_dict2[weight].pop()
            if len(weight_dict[weight]) == 0:
                del weight_dict[weight]
                del weight_dict2[weight]
                index = unique_weights[weight]
                last_weight = unique_weights2[-1]
                if weight != last_weight:
                    unique_weights2[index] = last_weight
                    unique_weights[last_weight] = index
                unique_weights2.pop()
                del unique_weights[weight]

    # Add new weights and links to the weight dictionaries and unique weights if necessary
    for i in range(len(new_weight)):
        weight = new_weight[i]
        edge = new_links[i]
        if weight not in weight_dict:
            weight_dict[weight] = Dict.empty(
                key_type=edge_type,
                value_type=node_type
            )
        if weight not in weight_dict2:
            weight_dict2[weight] = List.empty_list(edge_type)
        weight_dict2[weight].append(edge)
        
        weight_dict[weight][edge] = len(weight_dict2[weight]) - 1
        if weight not in unique_weights:
            unique_weights2.append(weight)
            unique_weights[weight] = len(unique_weights2) - 1
    return weight_dict, weight_dict2, unique_weights, unique_weights2


@conditional_njit
def update_adj_list(adj_list, adj_list2, old_links, new_links):
    """
    Update the adjacency lists when edges are added or removed.

    Args:
        adj_list (Dict[int, Dict[int, int]]): Dictionary mapping node to neighbors and indices in adj_list2.
        adj_list2 (Dict[int, List[int]]): Dictionary mapping node to list of neighbors.
        old_links (List[Tuple[int, int]]): List of edges to be removed.
        new_links (List[Tuple[int, int]]): List of edges to be added.

    Returns:
        adj_list (Dict[int, Dict[int, int]]): Updated adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Updated adjacency list dictionary.
    """
    # Remove old links from adjacency lists using swap-and-pop
    for i in range(len(old_links)):
        edge = old_links[i]
        u = edge[0]
        v = edge[1]
        

        index_v = adj_list[u][v]  
        index_u = adj_list[v][u] 
        
        vicini_u = adj_list2[u]
        last_neighbor_u = vicini_u[-1] 
        if index_v < len(vicini_u) - 1: 
            vicini_u[index_v] = last_neighbor_u 
            adj_list[u][last_neighbor_u] = index_v
        vicini_u.pop()  
        adj_list2[u] = vicini_u
        

        vicini_v = adj_list2[v]
        last_neighbor_v = vicini_v[-1]  
        if index_u < len(vicini_v) - 1:  
            vicini_v[index_u] = last_neighbor_v  
            adj_list[v][last_neighbor_v] = index_u
        vicini_v.pop()  
        adj_list2[v] = vicini_v
        
        
        del adj_list[u][v]
        del adj_list[v][u]

    # Add new links to the adjacency lists
    for i in range(len(new_links)):
        edge = new_links[i]
        u = edge[0]
        v = edge[1]
        vicini_u = adj_list2[u]
        vicini_v = adj_list2[v]
        vicini_u.append(v)
        vicini_v.append(u)
        adj_list2[u] = vicini_u
        adj_list2[v] = vicini_v
        adj_list[u][v] = len(vicini_u)-1
        adj_list[v][u] = len(vicini_v)-1

    return adj_list, adj_list2



@conditional_njit
def rectangle_random_search(A2, adj_list, adj_list2):
    """
    Randomly search for a 4-cycle (rectangle) in an undirected graph using neighbor path sampling.

    Args:
        A2 (np.ndarray): 2D square adjacency matrix of the graph.
        adj_list (Dict[int, Dict[int, int]]): Dictionary mapping node to neighbors and indices.
        adj_list2 (Dict[int, List[int]]): Dictionary mapping node to list of neighbors.

    Returns:
        rectangle (Tuple[int, int, int, int]): Tuple with 4 node IDs (u, u1, u2, u3), or (-1, -1, -1, -1) if no valid rectangle is found.
    """
    n = A2.shape[0]
    #random select a node finding its neighbors
    nodo = np.random.randint(0, n)
    vicini = adj_list2[nodo]
    #if the node has less than 2 neighbors it can't be part of a rectangle so the searching algorithm ends.
    if len(vicini) < 2:
        return (-1, -1, -1, -1)
    else:
        #random select 2 node amog the neighbors of "nodo" and take their neighbors.
        r1 = np.random.randint(0, len(vicini))
        r2 = np.random.randint(0, len(vicini))
        while r1 == r2:   
            r2 = np.random.randint(0, len(vicini))
        nodo2 = vicini[r2]
        nodo1 = vicini[r1]
        bol = np.random.rand()
        #randomly select one of the two neighbors and find its neighbors and call it "nodo3"
        if bol < 0.5:
            vicini_lista  = adj_list2[nodo1]
            vicini_comparison = adj_list[nodo2]
        else:
            vicini_lista  = adj_list2[nodo2]
            vicini_comparison = adj_list[nodo1]
        r3 = np.random.randint(0, len(vicini_lista))
        nodo3 = vicini_lista[r3]
        #if nodo3 is a neighbor of the remaining neighbor, the rectangle is found.
        if nodo3 in vicini_comparison and nodo3 != nodo:
            rectangle = (nodo, nodo1, nodo2, nodo3)
        else:
            rectangle = (-1, -1, -1, -1)

    return rectangle

@conditional_njit
def weight_shuffling(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2):
    """
    Perform weight shuffling on a 4-cycle (rectangle) of an undirected graph, preserving topology and strengths.

    Args:
        A2 (np.ndarray): 2D square adjacency matrix of the graph (modified in-place).
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Weight to edge-index dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Weight to edge-list dictionary.
        unique_weights (Dict[int, int]): Unique weight mapping.
        unique_weights2 (List[int]): Unique weights list.
        adj_list (Dict[int, Dict[int, int]]): Adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Adjacency list dictionary.

    Returns:
        A2 (np.ndarray): Updated adjacency matrix.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Updated weight dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Updated edge list dictionary by weight.
        unique_weights (Dict[int, int]): Updated unique weights dictionary.
        unique_weights2 (List[int]): Updated unique weights list.
    """
    n = A2.shape[0]

    # Select a random rectangle
    rectangle = rectangle_random_search(A2, adj_list, adj_list2)

    if rectangle[0] != -1:
        a = rectangle[0]
        b = rectangle[1]
        c = rectangle[2]
        d = rectangle[3]

        w11 = A2[a, b]
        w12 = A2[a, c]
        w21 = A2[b, d]
        w22 = A2[c, d]
        old_weight = np.array([w11, w22, w12, w21], dtype=np.int64)

        e1 = (a, b) if a < b else (b, a)
        e2 = (c, d) if c < d else (d, c)
        e3 = (a, c) if a < c else (c, a)
        e4 = (b, d) if b < d else (d, b)
        
        vet_link = [e1, e2, e3, e4]
        
        vet1 = np.array([w11, w22])
        vet2 = np.array([w12, w21])
    
        

        # Find all possible delta values and choose one uniformly at random
        min_aa = np.min(vet1)-1
        min_ab = (-1)*(np.min(vet2)-1)
        
        delta = np.random.randint(min_ab, min_aa+1)
        
        A2[a, b] = w11 - delta
        A2[b, a] = w11 - delta
        A2[c, d] = w22 - delta
        A2[d, c] = w22 - delta
        A2[a, c] = w12 + delta
        A2[c, a] = w12 + delta
        A2[b, d] = w21 + delta
        A2[d, b] = w21 + delta

        new_weight = np.array([A2[a, b], A2[c, d], A2[a, c], A2[b, d]], dtype=np.int64)
        weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, vet_link, vet_link)
        return A2, weight_dict, weight_dict2, unique_weights, unique_weights2
    else:
        return A2, weight_dict, weight_dict2, unique_weights, unique_weights2

@conditional_njit
def simple_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2):
    """
    Swap two edges of the same weight in an undirected graph, preserving degree sequence and edge weights.

    Args:
        A2 (np.ndarray): 2D square adjacency matrix of the graph.
        adj_list (Dict[int, Dict[int, int]]): Adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Adjacency list dictionary.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Weight to edge-index dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Weight to edge-list dictionary.
        unique_weights (Dict[int, int]): Unique weight mapping.
        unique_weights2 (List[int]): Unique weights list.

    Returns:
        A2 (np.ndarray): Updated adjacency matrix.
        adj_list (Dict[int, Dict[int, int]]): Updated adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Updated adjacency list dictionary.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Updated weight dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Updated edge list dictionary by weight.
        unique_weights (Dict[int, int]): Updated unique weights dictionary.
        unique_weights2 (List[int]): Updated unique weights list.
    """
    # Randomly select a weight among the unique weights in the current graph
    n = A2.shape[0]
    unique_weights2_len = len(unique_weights2)
    r = np.random.randint(0, unique_weights2_len)
    weight = unique_weights2[r]
    edges = weight_dict2[weight]
    k = len(edges)
    if k < 2:
        return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2
    # Randomly select two edges with the same weight
    i = np.random.randint(0, k)
    j = np.random.randint(0, k)
    if i != j:
        edge1 = edges[i]
        edge2 = edges[j]
        u1 = edge1[0]
        u2 = edge2[0]
        v1 = edge1[1]
        v2 = edge2[1]
        # Check if the four nodes are distinct, if not the swap is not possible
        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2
        r = np.random.rand()
        if r < 0.5:
            a, l = u1, v2
            b, m = v1, u2
        else:
            a, l = u1, u2
            b, m = v1, v2
        # Check if the swap is possible without creating self-loops or multi-edges
        if A2[a, l] == 0 and A2[b, m] == 0:
            # Perform the swap and update the data structures
            A2[u1, v1] = 0
            A2[u2, v2] = 0
            A2[v1, u1] = 0
            A2[v2, u2] = 0
            A2[a, l] = weight
            A2[b, m] = weight
            A2[l, a] = weight
            A2[m, b] = weight
            old_weight = np.array([weight, weight], dtype=np.int64)
            if u1 < v1:
                old_link1 = (u1, v1)
            else:
                old_link1 = (v1, u1)
            if u2 < v2:
                old_link2 = (u2, v2)
            else:
                old_link2 = (v2, u2)
            old_links = [old_link1, old_link2]
            new_weight = np.array([weight, weight], dtype=np.int64)
            if a < l:
                new_link1 = (a, l)
            else:
                new_link1 = (l, a)
            if b < m:
                new_link2 = (b, m)
            else:
                new_link2 = (m, b)
            new_links = [new_link1, new_link2]

            adj_list, adj_list2 = update_adj_list(adj_list, adj_list2, old_links, new_links)
            weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, old_links, new_links)

    return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2



@conditional_njit
def research_five_nodes_structure(A2, adj_list2):
    """
    Search for a 5-node path/structure in the undirected graph using random path sampling.

    Args:
        A2 (np.ndarray): 2D square adjacency matrix of the graph.
        adj_list2 (Dict[int, List[int]]): Adjacency list mapping each node to its neighbors.

    Returns:
        structure (List[int]): List of 5 node IDs representing the chain found, or fewer than 5 nodes if not found.
    """
    n = A2.shape[0]
    structure = List.empty_list(node_type)
    old_node = -1
    for i in range(5):
        if i == 0:
            node = np.random.randint(0, n)
            vicini = adj_list2[node]
        else:
            k = len(vicini)
            if k == 0:
                break
            if k < 2 and i != 1:
                break
            elif k == 2:
                if i != 1:
                    if vicini[0] in structure and vicini[1] in structure:
                        break
                    else:
                        if vicini[0] not in structure:
                            node = vicini[0]
                        else:
                            node = vicini[1]
                    if i == 1:
                        r = np.random.randint(0, 2)
                        node = vicini[r]
            else:
                while True:
                    r = np.random.randint(0, k)
                    node2 = vicini[r]
                    if node2 != old_node:
                        node = node2
                        break
                
        if node in structure:
            break
        else:
            vicini = adj_list2[node]
            structure.append(node)
            old_node = node
    
    return structure


@conditional_njit
def check_five_nodes_structure(A2, structure):
    """
    Check if a 5-node structure is suitable for bridge edge swap.

    Args:
        A2 (np.ndarray): 2D square adjacency matrix of the graph.
        structure (List[int]): List of 5 node IDs forming the candidate chain.

    Returns:
        check (bool): True if the structure satisfies the conditions for bridge edge swap, False otherwise.
    """
    n = A2.shape[0]
    check = False
    a = structure[0]
    b = structure[1]
    c = structure[2]
    d = structure[3]
    e = structure[4]
    if A2[a, d] == 0 and A2[b, e] == 0:
        # Check if the bridge (c, d) has enough weight to transfer to (b, c) to compensate the loss due to the swap
        if A2[a,b] > A2[e, d] and A2[c, d] > A2[a, b] - A2[e, d]:
            check = True
        if A2[e, d] > A2[a, b] and A2[b, c] > A2[e, d] - A2[a, b]:
            check = True
    return check



@conditional_njit
def bridge_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2):
    """
    Swap two edges with different weights using a 5-node bridge path, preserving degree sequence and strengths.

    Args:
        A2 (np.ndarray): 2D square adjacency matrix of the graph (modified in-place).
        adj_list (Dict[int, Dict[int, int]]): Adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Adjacency list dictionary.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Weight to edge-index dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Weight to edge-list dictionary.
        unique_weights (Dict[int, int]): Unique weight mapping.
        unique_weights2 (List[int]): Unique weights list.

    Returns:
        A2 (np.ndarray): Updated adjacency matrix.
        adj_list (Dict[int, Dict[int, int]]): Updated adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Updated adjacency list dictionary.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Updated weight dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Updated edge list dictionary by weight.
        unique_weights (Dict[int, int]): Updated unique weights dictionary.
        unique_weights2 (List[int]): Updated unique weights list.
    """
    n = A2.shape[0]
    
    structure = research_five_nodes_structure(A2, adj_list2)
    
    if len(structure) == 5:

        if check_five_nodes_structure(A2, structure):
            
            a = structure[0]
            b = structure[1]
            c = structure[2]
            d = structure[3]
            e = structure[4]
            w11 = A2[a, b]
            w22 = A2[d, e]
            delta = w11 - w22
            old_weights = np.array([w11, w22, A2[b, c], A2[c, d]], dtype=np.int64)
            A2[a, b] = 0
            A2[d, e] = 0
            A2[a, d] = w11
            A2[b, e] = w22
            A2[b, c] = A2[b, c] + delta
            A2[c, d] = A2[c, d] - delta

            A2[b, a] = A2[a, b]
            A2[e, d] = A2[d, e]
            A2[d, a] = A2[a, d]
            A2[e, b] = A2[b, e]
            A2[c, b] = A2[b, c]
            A2[d, c] = A2[c, d]


            e1 = (a, b) if a < b else (b, a)
            e2 = (d, e) if d < e else (e, d)
            e3 = (b, c) if b < c else (c, b)
            e4 = (c, d) if c < d else (d, c)

            old_links = [e1, e2, e3, e4]

            e1 = (a, d) if a < d else (d, a)
            e2 = (b, e) if b < e else (e, b)

            new_links = [e1, e2, e3, e4]
            new_weights = np.array([w11, w22, A2[b, c], A2[c, d]], dtype=np.int64)
                

            # Update data structures
            broken_links = [old_links[0], old_links[1]]
            created_links = [new_links[0], new_links[1]]
            adj_list, adj_list2 = update_adj_list(adj_list, adj_list2, broken_links, created_links)
            weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weights, new_weights, old_links, new_links)
    return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2

@conditional_njit
def pre_compute_wes(A):
    """
    Pre-compute data structures for the WES algorithm.

    Args:
        A (np.ndarray): 2D square adjacency matrix of the undirected graph.

    Returns:
        A2 (np.ndarray): Copy of the input adjacency matrix.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Mapping of weight to edge-to-index dict.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Mapping of weight to list of edges.
        adj_list (Dict[int, Dict[int, int]]): Mapping of node to neighbors and indices.
        adj_list2 (Dict[int, List[int]]): Mapping of node to list of neighbors.
        unique_weights (Dict[int, int]): Mapping of unique weight to index.
        unique_weights2 (List[int]): List of unique weights.
    """
    A2 = A.copy()
    adj_list, adj_list2 = from_adj_matrix_to_adj_list(A2)
    weight_dict, weight_dict2, unique_weights, unique_weights2 = get_weight_dict(A2)
    return A2, weight_dict, weight_dict2, adj_list, adj_list2, unique_weights, unique_weights2



@conditional_njit
def WES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations):
    """
    Randomize an undirected graph using the WES algorithm, preserving degree and weight sequences.

    Args:
        A2 (np.ndarray): 2D square adjacency matrix of the undirected graph (modified in-place).
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Mapping of weight to edge-to-index dict.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Mapping of weight to list of edges.
        unique_weights (Dict[int, int]): Mapping of unique weight to index.
        unique_weights2 (List[int]): List of unique weights.
        adj_list (Dict[int, Dict[int, int]]): Mapping of node to neighbors and indices.
        adj_list2 (Dict[int, List[int]]): Mapping of node to list of neighbors.
        n_iterations (int): Number of randomization iterations to perform.

    Returns:
        A2 (np.ndarray): The randomized adjacency matrix.
    """
    
    p = 1/3
    prob = np.array([p, p, p])
    prob = np.cumsum(prob)

    for it in range(n_iterations):
        r = np.random.rand()
        if r < prob[0]:
            A2, weight_dict, weight_dict2, unique_weights, unique_weights2 = weight_shuffling(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2)
        elif prob[0] <= r < prob[1]:
            A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2 = simple_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2)
        else:
            A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2 = bridge_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2)
    return A2









