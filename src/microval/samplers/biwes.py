import numpy as np
from numba import njit
import numba.types as nt
from numba.typed import Dict, List
from numba.types import int64, ListType, UniTuple




USE_NJIT = True  # Cambia questo valore per switchare

# Crea un decoratore condizionale
def conditional_njit(func):
    """Decoratore che applica njit solo se USE_NJIT è True"""
    if USE_NJIT:
        return njit(func)
    else:
        return func




node_type = nt.int64
neighbors_type = nt.DictType(node_type, node_type)
edge_type = nt.UniTuple(node_type, 2)
edge_list_dict_type = nt.DictType(edge_type, node_type)
list_node_type = ListType(node_type)
edge_list_ty = ListType(edge_type)




@conditional_njit
def from_adj_matrix_to_adj_list(A):
    """
    Create two adjacency lists representation for a bipartite graph from its bi-adjacency matrix.

    Args:
        A (np.ndarray): 2D bi-adjacency matrix of shape (na, nb) of the bipartite graph.

    Returns:
        adj_list (Dict[int, Dict[int, int]]): Dictionary mapping each node to a dictionary of its neighbors and their indices in adj_list2.
        adj_list2 (Dict[int, List[int]]): Dictionary mapping each node to a list of its neighboring node IDs.
    """
    na, nb = A.shape[0], A.shape[1]
    adj_list = Dict.empty(
        key_type=node_type,
        value_type=neighbors_type
    )
    
    adj_list2 = Dict.empty(
        key_type=node_type,
        value_type=list_node_type
    )

    for i in range(na):
        neighbors = Dict.empty(
            key_type=node_type,
            value_type=node_type
        )
        cont = 0
        vicini = List.empty_list(node_type)
        for j in range(nb):
            if A[i, j] != 0:
                neighbors[j + na] = cont
                vicini.append(j + na)
                cont += 1
        adj_list[i] = neighbors
        adj_list2[i] = vicini
    
    for j in range(nb):
        neighbors = Dict.empty(
            key_type=node_type,
            value_type=node_type
        )
        cont = 0
        vicini = List.empty_list(node_type)
        for i in range(na):
            if A[i, j] != 0:
                neighbors[i] = cont
                vicini.append(i)
                cont += 1
        adj_list[j + na] = neighbors
        adj_list2[j + na] = vicini


    return adj_list, adj_list2

@conditional_njit
def get_weight_dict(A):
    """
    Construct weight dictionaries and unique weight trackers from a bi-adjacency matrix.

    Args:
        A (np.ndarray): 2D bi-adjacency matrix of shape (na, nb) of the bipartite graph.

    Returns:
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Dictionary mapping each weight to a dictionary of (edge -> index in weight_dict2).
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Dictionary mapping each weight to a list of edges with that weight.
        unique_weights (Dict[int, int]): Dictionary mapping each unique weight to its index in unique_weights2.
        unique_weights2 (List[int]): List of unique edge weights present in the graph.
    """
    na, nb= A.shape[0], A.shape[1]
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
    for i in range(na):
        for j in range(nb):
            if A[i, j] != 0:
                weight = A[i, j]
                edge = (i, j + na)
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


rect_type = nt.UniTuple(node_type, 4)
list_dict_rect = nt.DictType(rect_type, node_type)


@conditional_njit
def update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, old_links, new_links):
    """
    Update weight dictionaries and unique weights when edges change weights or are replaced.

    Args:
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Mapping of weight to dict of (edge -> index).
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Mapping of weight to list of edges.
        unique_weights (Dict[int, int]): Mapping of unique weight to its index.
        unique_weights2 (List[int]): List of unique weights.
        old_weight (np.ndarray): Array of old edge weights to remove or decrease.
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

    # Add new weights
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
def rectangle_random_search(A2, adj_list, adj_list2):
    """
    Randomly search for a 4-cycle (rectangle) in the bipartite graph using neighbor sampling.

    Args:
        A2 (np.ndarray): 2D bi-adjacency matrix of the bipartite graph.
        adj_list (Dict[int, Dict[int, int]]): Dictionary mapping node to neighbors and indices.
        adj_list2 (Dict[int, List[int]]): Dictionary mapping node to list of neighbors.

    Returns:
        rect (Tuple[int, int, int, int]): Tuple with node IDs of the rectangle (u1, u2, v1, v2), or (-1, -1, -1, -1) if no valid rectangle is found.
    """
    na = A2.shape[0]
    nb = A2.shape[1]
    nodo = np.random.randint(0, na+nb)
    vicini = adj_list2[nodo]
    if len(vicini) < 2:
        return (-1,-1,-1,-1)
    else:
        r1 = np.random.randint(0, len(vicini))
        r2 = np.random.randint(0, len(vicini))
        while r1 == r2:   
            r2 = np.random.randint(0, len(vicini))
        nodo2 = vicini[r2]
        nodo1 = vicini[r1]
        vicini1 = adj_list[nodo1]
        vicini2 = adj_list[nodo2]
        bol = np.random.rand()
        if bol < 0.5:
            vicini_lista  = adj_list2[nodo1]
            vicini_comparison = adj_list[nodo2]
        else:
            vicini_lista  = adj_list2[nodo2]
            vicini_comparison = adj_list[nodo1]
        r3 = np.random.randint(0, len(vicini_lista))
        nodo3 = vicini_lista[r3]
        if nodo3 in vicini_comparison and nodo3 != nodo:
            if nodo < na:
                u1 = nodo
                u2 = nodo3
                v1 = nodo1
                v2 = nodo2
            else:
                u1 = nodo1
                u2 = nodo2
                v1 = nodo
                v2 = nodo3
            if u1 < u2 and v1 < v2:
                rect = (u1, u2, v1, v2)
            elif u1 < u2 and v1 > v2:
                rect = (u1, u2, v2, v1)
            elif u1 > u2 and v1 < v2:
                rect = (u2, u1, v1, v2)
            else:
                rect = (u2, u1, v2, v1)
            rect = (u1, u2, v1, v2)
            return rect
        else:
            return (-1,-1,-1,-1)
@conditional_njit
def weight_shuffling(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2):
    """
    Perform weight shuffling on a 4-cycle (rectangle) of the bipartite graph, preserving topology and strengths.

    Args:
        A2 (np.ndarray): 2D bi-adjacency matrix of the bipartite graph (modified in-place).
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Weight to edge-index dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Weight to edge-list dictionary.
        unique_weights (Dict[int, int]): Unique weight mapping.
        unique_weights2 (List[int]): Unique weights list.
        adj_list (Dict[int, Dict[int, int]]): Adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Adjacency list dictionary.

    Returns:
        A2 (np.ndarray): The updated bi-adjacency matrix.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): The updated weight dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): The updated edge list dictionary by weight.
        unique_weights (Dict[int, int]): The updated unique weights dictionary.
        unique_weights2 (List[int]): The updated unique weights list.
    """
    na = A2.shape[0]

    # Select a random rectangle
    rectangle = rectangle_random_search(A2, adj_list, adj_list2)

    if rectangle[0] != -1:
        u1 = rectangle[0]
        u2 = rectangle[1]
        v1 = rectangle[2]
        v2 = rectangle[3]
        w11 = A2[u1, v1-na]
        w12 = A2[u1, v2-na]
        w21 = A2[u2, v1-na]
        w22 = A2[u2, v2-na]
        old_weight = np.array([w11, w22, w12, w21], dtype=np.int64)
        vet_link = [(u1, v1), (u2, v2), (u1, v2), (u2, v1)]
        involved_nodes = np.array([u1, u2], dtype=np.int64)
        vet1 = np.array([w11, w22])
        vet2 = np.array([w12, w21])
    
        # Find all possible delta values and choose one uniformly at random
        min_aa = np.min(vet1)-1
        min_ab = (-1)*(np.min(vet2)-1)
        
        delta = np.random.randint(min_ab, min_aa+1)
        
        A2[u1, v1-na] = w11 - delta
        A2[u2, v2-na] = w22 - delta
        A2[u1, v2-na] = w12 + delta
        A2[u2, v1-na] = w21 + delta

        new_weight = np.array([A2[u1, v1-na], A2[u2, v2-na], A2[u1, v2-na], A2[u2, v1-na]], dtype=np.int64)
        weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, vet_link, vet_link)
        

    return A2, weight_dict, weight_dict2, unique_weights, unique_weights2



@conditional_njit
def update_adj_list(A2, adj_list, adj_list2, old_links, new_links):
    """
    Update the adjacency lists when edges are removed or added.

    Args:
        A2 (np.ndarray): 2D bi-adjacency matrix.
        adj_list (Dict[int, Dict[int, int]]): Mapping of node to neighbors and index in adj_list2.
        adj_list2 (Dict[int, List[int]]): Mapping of node to list of neighbors.
        old_links (List[Tuple[int, int]]): List of edges to be removed.
        new_links (List[Tuple[int, int]]): List of edges to be added.

    Returns:
        adj_list (Dict[int, Dict[int, int]]): The updated adjacency index dictionary.
    """
    for i in range(len(old_links)):
        edge = old_links[i]
        u = edge[0]
        v = edge[1]
        
        # Get indices before removing
        index_v = adj_list[u][v]  # Index of v in neighbor list of u
        index_u = adj_list[v][u]  # Index of u in neighbor list of v
        
        # Remove from adj_list2 for u
        vicini_u = adj_list2[u]
        last_neighbor_u = vicini_u[-1]  # Save last neighbor
        if index_v < len(vicini_u) - 1:  # If not already the last
            vicini_u[index_v] = last_neighbor_u  # Replace with the last
            adj_list[u][last_neighbor_u] = index_v
        vicini_u.pop()  # Remove last
        adj_list2[u] = vicini_u
        
        # Remove from adj_list2 for v
        vicini_v = adj_list2[v]
        last_neighbor_v = vicini_v[-1]  # Save last neighbor
        if index_u < len(vicini_v) - 1:  # If not already the last
            vicini_v[index_u] = last_neighbor_v  # Replace with the last
            adj_list[v][last_neighbor_v] = index_u
        vicini_v.pop()  # Remove last
        adj_list2[v] = vicini_v
        
        # Remove from adj_list
        del adj_list[u][v]
        del adj_list[v][u]

    # Add new links
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
    return adj_list



@conditional_njit
def simple_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2):
    """
    Swap two edges of the same weight in a bipartite graph, preserving topology degree sequence and weights.

    Args:
        A2 (np.ndarray): 2D bi-adjacency matrix of the bipartite graph.
        adj_list (Dict[int, Dict[int, int]]): Adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Adjacency list dictionary.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Weight to edge-index dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Weight to edge-list dictionary.
        unique_weights (Dict[int, int]): Unique weight mapping.
        unique_weights2 (List[int]): Unique weights list.

    Returns:
        A2 (np.ndarray): Updated bi-adjacency matrix.
        adj_list (Dict[int, Dict[int, int]]): Updated adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Updated adjacency list dictionary.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Updated weight dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Updated edge list dictionary by weight.
        unique_weights (Dict[int, int]): Updated unique weights dictionary.
        unique_weights2 (List[int]): Updated unique weights list.
    """
    # Check if at least one move is possible
    na = A2.shape[0]
    unique_weights2_len = len(unique_weights2)
    r = np.random.randint(0, unique_weights2_len)
    weight = unique_weights2[r]
    edges = weight_dict2[weight]
    k = len(edges)
    if k < 2:
        return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2
    
    i = np.random.randint(0, k)
    j = np.random.randint(0, k)
    if i != j:
        edge1 = edges[i]
        edge2 = edges[j]
        u1 = edge1[0]
        u2 = edge2[0]
        v1 = edge1[1]
        v2 = edge2[1]
        if u1 != u2 and v1 != v2 and A2[u1, v2-na] == 0 and A2[u2, v1-na] == 0:
            A2[u1, v1-na] = 0
            A2[u2, v2-na] = 0
            A2[u1, v2-na] = weight
            A2[u2, v1-na] = weight
            old_weight = np.array([weight, weight], dtype=np.int64)
            old_links = [(u1, v1), (u2, v2)]
            new_weight = np.array([weight, weight], dtype=np.int64)
            new_links = [(u1, v2), (u2, v1)]

            adj_list = update_adj_list(A2, adj_list, adj_list2, old_links, new_links)
            weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, old_links, new_links)

    return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2





@conditional_njit
def research_five_nodes_structure(A2, adj_list2):
    """
    Search for a 5-node alternating chain in the bipartite graph via random path sampling.

    Args:
        A2 (np.ndarray): 2D bi-adjacency matrix of the bipartite graph.
        adj_list2 (Dict[int, List[int]]): Adjacency list mapping each node to its neighbors.

    Returns:
        structure (List[int]): List of 5 node IDs forming the chain, or fewer than 5 nodes if not found.
    """
    na = A2.shape[0]
    nb = A2.shape[1]
    n = na+ nb
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
                '''while True:
                    r = np.random.randint(0, k)
                    node = vicini[r]
                    if node not in structure:
                        break'''
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

    if len(structure) == 5:
        if structure[0] < na:
            temp = structure[1]
            structure[1] = structure[4]
            structure[4] = temp
            temp = structure[2]
            structure[2] = structure[4]
            structure[4] = temp
        else:
            temp = structure[0]
            structure[0] = structure[1]
            structure[1] = temp
            temp = structure[3]
            structure[3] = structure[4]
            structure[4] = temp
            temp = structure[1]
            structure[1] = structure[4]
            structure[4] = temp
            temp = structure[2]
            structure[2] = structure[4]
            structure[4] = temp

    return structure


@conditional_njit
def check_five_nodes_structure(A2, structure):
    """
    Check if a 5-node structure is suitable for performing a complex edge swap.

    Args:
        A2 (np.ndarray): 2D bi-adjacency matrix of the bipartite graph.
        structure (List[int]): List of 5 node IDs forming the candidate chain.

    Returns:
        check (bool): True if the structure satisfies the conditions for complex edge swap, False otherwise.
    """
    na = A2.shape[0]
    check = False
    if structure[4] < na:
        u1 = structure[0]
        u2 = structure[1]
        v1 = structure[2]
        v2 = structure[3]
        u3 = structure[4]
        if A2[u1, v2-na] == 0 and A2[u2, v1-na] == 0 and A2[u1, v1-na] != A2[u2, v2-na]:
            arr = np.array([A2[u1, v1-na], A2[u2, v2-na]])
            peso_max = np.max(arr)
            if A2[u1, v1-na] == peso_max:
                if A2[u3, v2-na] > np.abs(A2[u2, v2-na] - A2[u1, v1-na]):
                    check = True
            if A2[u2, v2-na] == peso_max:
                if A2[u3, v1-na] > np.abs(A2[u2, v2-na] - A2[u1, v1-na]):
                    check = True
    if structure[4] >= na:
        u1 = structure[0]
        u2 = structure[1]
        v1 = structure[2]
        v2 = structure[3]
        v3 = structure[4]
        if A2[u1, v2-na] == 0 and A2[u2, v1-na] == 0 and A2[u1, v1-na] != A2[u2, v2-na]:
            arr = np.array([A2[u1, v1-na], A2[u2, v2-na]])
            peso_max = np.max(arr)
            if A2[u1, v1-na] == peso_max:
                if A2[u2, v3-na] > np.abs(A2[u2, v2-na] - A2[u1, v1-na]):
                    check = True
            if A2[u2, v2-na] == peso_max:
                if A2[u1, v3-na] > np.abs(A2[u2, v2-na] - A2[u1, v1-na]):
                    check = True
    return check

@conditional_njit
def complex_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2):
    """
    Swap two edges of different weights using a 5-node bridge structure to preserve strength sequence.

    Args:
        A2 (np.ndarray): 2D bi-adjacency matrix of the bipartite graph.
        adj_list (Dict[int, Dict[int, int]]): Adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Adjacency list dictionary.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Weight to edge-index dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Weight to edge-list dictionary.
        unique_weights (Dict[int, int]): Unique weight mapping.
        unique_weights2 (List[int]): Unique weights list.

    Returns:
        A2 (np.ndarray): Updated bi-adjacency matrix.
        adj_list (Dict[int, Dict[int, int]]): Updated adjacency index dictionary.
        adj_list2 (Dict[int, List[int]]): Updated adjacency list dictionary.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Updated weight dictionary.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Updated edge list dictionary by weight.
        unique_weights (Dict[int, int]): Updated unique weights dictionary.
        unique_weights2 (List[int]): Updated unique weights list.
    """
    na = A2.shape[0]
    
    structure = research_five_nodes_structure(A2, adj_list2)
    
    if len(structure) == 5:
        #print(structure)
        if check_five_nodes_structure(A2, structure):
            

            na = A2.shape[0]
            u1 = structure[0]
            u2 = structure[1]
            v1 = structure[2]
            v2 = structure[3]
            w11 = A2[u1, v1-na]
            w22 = A2[u2, v2-na]
            delta = np.abs(w11 - w22)
            A2[u1, v1-na] = 0
            A2[u2, v2 - na] = 0

            if structure[4] < na:
                u3 = structure[4]
                old_weights = np.array([w11, w22, A2[u3, v1 - na], A2[u3, v2 - na]], dtype=np.int64)
                A2[u1, v2 - na] = w11
                A2[u2, v1 - na] = w22
                if w11 > w22:
                    A2[u3, v1 - na] = A2[u3, v1 - na] + delta
                    A2[u3, v2 - na] = A2[u3, v2 - na] - delta
                else:
                    A2[u3, v1 - na] = A2[u3, v1 - na] - delta
                    A2[u3, v2 - na] = A2[u3, v2 - na] + delta
                old_links = [(u1, v1), (u2, v2), (u3, v1), (u3, v2)]
                new_links = [(u1, v2), (u2, v1), (u3, v1), (u3, v2)]
                new_weights = np.array([w11, w22, A2[u3, v1 - na], A2[u3, v2 - na]], dtype=np.int64)
                involved_nodes_special = np.array([u1, u2, u3], dtype=np.int64)
                    
            if structure[4] >= na:
                v3 = structure[4]
                old_weights = np.array([w11, w22, A2[u1, v3 - na], A2[u2, v3 - na]], dtype=np.int64)
                A2[u1, v2 - na] = w22
                A2[u2, v1 - na] = w11
                if w11 > w22:
                    A2[u1, v3 - na] = A2[u1, v3 - na] + delta
                    A2[u2, v3 - na] = A2[u2, v3 - na] - delta
                else:
                    A2[u1, v3 - na] = A2[u1, v3 - na] - delta
                    A2[u2, v3 - na] = A2[u2, v3 - na] + delta
                old_links = [(u1, v1), (u2, v2), (u1, v3), (u2, v3)]
                new_links = [(u1, v2), (u2, v1), (u1, v3), (u2, v3)]
                new_weights = np.array([w22, w11, A2[u1, v3 - na], A2[u2, v3 - na]], dtype=np.int64)
                involved_nodes_special = np.array([u1, u2], dtype=np.int64)
            # Update data structures

            broken_links = [(u1, v1), (u2, v2)]
            created_links = [(u1, v2), (u2, v1)]
            adj_list = update_adj_list(A2, adj_list, adj_list2, broken_links, created_links)
            weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weights, new_weights, old_links, new_links)

            
            #print("Complex edge swap performed")
    return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2


@conditional_njit
def pre_compute_biwes(A):
    """
    Pre-compute data structures for the BiWES algorithm.

    Args:
        A (np.ndarray): 2D bi-adjacency matrix of the bipartite graph.

    Returns:
        A2 (np.ndarray): Copy of the input bi-adjacency matrix.
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
def BiWES(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations):
    """
    Randomize a bipartite graph using the BiWES algorithm, preserving degree and weight sequences.

    Args:
        A2 (np.ndarray): 2D bi-adjacency matrix of the bipartite graph (modified in-place).
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): Mapping of weight to edge-to-index dict.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): Mapping of weight to list of edges.
        unique_weights (Dict[int, int]): Mapping of unique weight to index.
        unique_weights2 (List[int]): List of unique weights.
        adj_list (Dict[int, Dict[int, int]]): Mapping of node to neighbors and indices.
        adj_list2 (Dict[int, List[int]]): Mapping of node to list of neighbors.
        n_iterations (int): Number of randomization iterations to perform.

    Returns:
        A2 (np.ndarray): The randomized bi-adjacency matrix.
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
            # Complex edge swap
            A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2 = complex_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2)
    return A2
