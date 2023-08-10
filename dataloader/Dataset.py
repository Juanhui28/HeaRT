"""
constructing the hashed data objects used by elph and buddy
"""

import os
from time import time

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce
import scipy.sparse as ssp
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.heuristics import RA
from src.utils import ROOT_DIR, get_src_dst_degree, get_pos_neg_edges, get_same_source_negs
from src.hashing import ElphHashes

# TODO: we do not support only test on a subset
class LPDataset(Dataset):
    def __init__(
            self, root, split, data, pos_edges, neg_edges, use_coalesce=False,
            directed=False, **kwargs):
        self.data = data
        self.split = split  # string: train, valid or test
        self.root = root
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.use_coalesce = use_coalesce
        self.directed = directed
        super(LPDataset, self).__init__(root)
        self.x = data.x

        if self.use_coalesce:  # compress multi-edge into edge with weight
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index, data.edge_weight,
                data.num_nodes, data.num_nodes)

        if 'edge_weight' in data:
            self.edge_weight = data.edge_weight.view(-1)
        else:
            self.edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        if self.directed:  # make undirected graphs like citation2 directed
            print(
                f'this is a directed graph. Making the adjacency matrix undirected to propagate features and calculate subgraph features')
            self.edge_index, self.edge_weight = to_undirected(data.edge_index, self.edge_weight)
        else:
            self.edge_index = data.edge_index
        self.A = ssp.csr_matrix(
            (self.edge_weight, (self.edge_index[0], self.edge_index[1])),
            shape=(data.num_nodes, data.num_nodes)
        )

        self.degrees = torch.tensor(self.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()

    def preprocess_node_feature(data, args):
        pass

    def process_node_feature(data, args):
        pass
    
    def process_structure_feature(data, args):
        pass

    def process_subgraph_feature(data, args):
        pass






