import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from fast_pagerank import pagerank_power

def PPR(A, edge_index):
    """
    The Personalized PageRank heuristic score.
    Need to install fast_pagerank by "pip install fast-pagerank"
    Too slow for large datasets now.
    :param A: A CSR matrix using the 'message passing' edges
    :param edge_index: The supervision edges to be scored
    :return:
    """
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[:, 0])
    dst_index = edge_index[sort_indices, 1]
    edge_reindex = torch.stack([src_index, dst_index])
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_reindex.shape[1])):
        if i < j:
            continue
        src = edge_reindex[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        # get the ppr for the current source node
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        # get ppr for all links that start at this source to save recalculating the ppr score
        while edge_reindex[0, j] == src:
            j += 1
            if j == edge_reindex.shape[1]:
                break
        all_dst = edge_reindex[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    print(f'evaluated PPR for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_reindex
