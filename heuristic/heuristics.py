

import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from torch_sparse import spspmm
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
import pdb
import networkx as nx
import math
import concurrent.futures


def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
        # print('max cn: ', np.concatenate(scores, 0).max())

    return torch.FloatTensor(np.concatenate(scores, 0))


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    
    return torch.FloatTensor(scores)

def RA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / (A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Resource Allocation for {len(scores)} edges')
    return torch.FloatTensor(scores)

def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)

def shortest_path_parallel(graph, edge_index, max_length):
    # BFS shortest path function with a maximum length constraint
    def bfs_shortest_path_length(graph, start, goal, max_length):
        if start == goal:
            return 0

        explored = set()
        queue = [(start, 0)]

        while queue:
            current_node, length = queue.pop(0)
            if current_node not in explored:
                explored.add(current_node)
                neighbors = graph[current_node]

                for neighbor in neighbors:
                    if neighbor == goal:
                        return length + 1
                    else:
                        if length + 1 < max_length:
                            queue.append((neighbor, length + 1))

        return None

    # import ipdb
    # ipdb.set_trace()
    
    pairs = edge_index.t().tolist()
    results = {}
    scores = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:

        future_to_pair = {executor.submit(bfs_shortest_path_length, graph, pair[0], pair[1], max_length): pair for pair in pairs}

        for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(pairs), desc='calculating paths') :
            pair = future_to_pair[future]
            try:
                path_length = future.result()
                results[pair] = path_length
                scores.append(path_length)
            except Exception as e:
                print(f"Error occurred for pair {pair}: {e}")
                scores.append(999)
    
    return torch.FloatTensor(scores)


def shortest_path(A, edge_index, remove=False):
    
    scores = []
    G = nx.from_scipy_sparse_matrix(A)
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0
    print('remove: ', remove)
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        # if (s,t) in train_pos_list: train_pos_list.remove((s,t))
        # if (t,s) in train_pos_list: train_pos_list.remove((t,s))


        # G = nx.Graph(train_pos_list)
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):

            sp = nx.shortest_path_length(G, source=s, target=t)
            # if sp == 0:
            #     print(1)
        else:
            sp = 999
        

        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
    
        # import ipdb
        # ipdb.set_trace()
        scores.append(1/(sp))
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores)

def katz_apro(A, edge_index, beta=0.005, path_len=3, remove=False):
    scores = []
    G = nx.from_scipy_sparse_matrix(A)
    path_len = int(path_len)
    count = 0
    add_flag1 = 0
    add_flag2 = 0
    count1 = count2 = 0
    betas = np.zeros(path_len)
    print('remove: ', remove)
    for i in range(len(betas)):
        betas[i] = np.power(beta, i+1)
    
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        if s == t:
            count += 1
            scores.append(0)
            continue
        
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
                
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1


        paths = np.zeros(path_len)
        for path in nx.all_simple_paths(G, source=s, target=t, cutoff=path_len):
            paths[len(path)-2] += 1  
        
        kz = np.sum(betas * paths)

        scores.append(kz)
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
        
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores)


def katz_close(A, edge_index, beta=0.005, path_len=3, remove=False):

    scores = []
    G = nx.from_scipy_sparse_matrix(A)

    adj = nx.adjacency_matrix(G, nodelist=range(len(G.nodes)))
    aux = adj.T.multiply(-beta).todense()
    np.fill_diagonal(aux, 1+aux.diagonal())
    sim = np.linalg.inv(aux)
    np.fill_diagonal(sim, sim.diagonal()-1)

    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        scores.append(sim[s,t])

    
    return torch.FloatTensor(scores)

def get_degree(degree_tensor, edge_index, batch_size=10000000):
    scores = []
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        src_degree, dst_degree = degree_tensor[src], degree_tensor[dst]
        scores.append(torch.reshape(torch.stack([src_degree, dst_degree]), (edge_index.size(1), 2)))
    return scores[0]
