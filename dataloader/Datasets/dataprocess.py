"""
Read and split ogb and planetoid datasets
"""

import os
import time

import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   to_undirected)
from torch_geometric.loader import DataLoader as pygDataLoader

from torch_sparse import SparseTensor
# from src.utils import ROOT_DIR, get_same_source_negs
# from src.lcc import get_largest_connected_component, remap_edges, get_node_mapper
# from src.datasets.seal import get_train_val_test_datasets
# from src.datasets.elph import get_hashed_train_val_test_datasets, make_train_eval_data

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")



def filter_by_year(data, split_edge, year):
    """
    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


def get_split_edge_citation2(split_edge):

    source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
    pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0).t()  ###[num_edge,2]

    source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
    pos_valid_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0).t() ###[num_edge,2]
    val_neg_edge = split_edge['valid']['target_node_neg'] 

    neg_valid_edge = torch.stack([source.repeat_interleave(val_neg_edge.size(1)), 
                            split_edge['valid']['target_node_neg'] .view(-1)]).t() ###[num_edge,2]

    source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
    pos_test_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0).t() ###[num_edge,2]
    test_neg_edge = split_edge['test']['target_node_neg']

    neg_test_edge = torch.stack([source.repeat_interleave(test_neg_edge.size(1)), 
                            test_neg_edge.view(-1)]).t()  ###[num_edge,2]
    
    split_edge['train']['edge'] = pos_train_edge
    split_edge['valid']['edge'] = pos_valid_edge
    split_edge['valid']['edge_neg'] = neg_valid_edge
    
    split_edge['test']['edge'] = pos_test_edge
    split_edge['test']['edge_neg'] = neg_test_edge

    return split_edge


def wrap_data( x, edge_index, edge_weight, split_edge,args,num_nodes, data=None):

    train_pos = split_edge['train']['edge']
    train_neg = split_edge['train']['edge_neg']

    train_val = split_edge['train_val']['edge']
    valid_neg = split_edge['train_val']['edge_neg'] 

    valid_pos = split_edge['valid']['edge']
    valid_neg = split_edge['valid']['edge_neg'] 
    test_pos = split_edge['test']['edge'] 
    test_neg  = split_edge['test']['edge_neg'] 
    splits = {}

    edge_label = torch.cat([torch.ones(len(train_pos)), torch.zeros(len(train_pos))], dim=0)
    edge_label_index = torch.cat([train_pos, train_neg], dim=0).t()
    splits['train'] = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                        edge_label_index=edge_label_index)
    if args.use_train_val:
        edge_label = torch.cat([torch.ones(len(train_val)), torch.zeros(len(valid_neg))], dim=0)
        edge_label_index = torch.cat([train_val, valid_neg], dim=0).t()
        splits['train_val'] = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                        edge_label_index=edge_label_index)
    
    
    edge_label = torch.cat([torch.ones(len(valid_pos)), torch.zeros(len(valid_neg))], dim=0)
    edge_label_index = torch.cat([valid_pos, valid_neg], dim=0).t()
    splits['valid'] = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                        edge_label_index=edge_label_index)
    
    edge_label = torch.cat([torch.ones(len(test_pos)), torch.zeros(len(test_neg))], dim=0)
    edge_label_index = torch.cat([test_pos, test_neg], dim=0).t()
    splits['test'] = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                        edge_label_index=edge_label_index)
    
    if args.use_valedges_as_input:
        vei, vw = to_undirected(valid_pos.t(), split_edge['valid']['weight'])
        full_edge_index = torch.cat([data.edge_index, vei], dim=1)
        full_edge_weight = torch.cat([data.edge_weight, vw.unsqueeze(-1)], dim=0)
        
        full_adj_t = SparseTensor.from_edge_index(full_edge_index, full_edge_weight.view(-1), [num_nodes, num_nodes])
        splits['test'] = Data(x=x, edge_index=full_edge_index, edge_weight=full_edge_weight, edge_label=edge_label,
                        edge_label_index=edge_label_index)
            
    else:
        full_edge_index = None
        full_edge_weight = None
        full_adj_t = None

   

    return splits, full_edge_index, full_edge_weight, full_adj_t



def get_ogb_data(data, split_edge, dataset_name, use_train_val):
    """
    ogb datasets come with fixed train-val-test splits and a fixed set of negatives against which to evaluate the test set
    The dataset.data object contains all of the nodes, but only the training edges
    @param dataset:
    @param use_valedges_as_input:
    @return:
    """
    save_data_name = dataset_name.replace('-', '_')
    # if num_negs == 1:
    negs_name = f'{ROOT_DIR}/dataset/{save_data_name}/train_neg_edge'
  
    print(f'looking for negative edges at {negs_name}')
    if os.path.exists(negs_name):
        print('loading negatives from disk')
        train_negs = torch.load(negs_name)
    else:
        print('negatives not found on disk. Generating negatives')
        train_negs = get_train_neg_edges(data, split_edge, dataset_name)
        torch.save(train_negs, negs_name)

    if 'citation2' in dataset_name:
        split_edge = get_split_edge_citation2(split_edge)
    
    split_edge['train']['edge_neg'] = train_negs
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]

    if use_train_val:
        train_val = split_edge['train']['edge'][idx]
        split_edge['train_val'] = dict()
        split_edge['train_val']['edge'] = train_val
        split_edge['train_val']['edge_neg'] = split_edge['valid']['edge_neg']


    return split_edge, idx



def get_train_neg_edges(data, split_edge, data_name):

    if 'citation2' not in data_name:
    
        pos_train = split_edge['train']['edge']

        edge_index = data.edge_index
        num_nodes = data.num_nodes
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
                    new_edge_index, num_nodes=num_nodes,
                    num_neg_samples=pos_train.size(0))

    else:
        num_nodes = data.num_nodes

        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0)

        dst_neg = torch.randint(0, num_nodes, (1, pos_train_edge.size(1)), dtype=torch.long)
        src_neg = pos_train_edge[0].repeat_interleave(1)
        neg_edge = torch.cat([src_neg.unsqueeze(0), dst_neg], dim=0)

    return neg_edge

