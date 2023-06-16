import os 
import random
import torch
import numpy as np

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected

import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset


def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"


def get_data_ogb(args):
    """
    Get data for OGB splits
    """
    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]

    split_edge = dataset.get_edge_split()

    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        edge_weight = data.edge_weight.to(torch.float)
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    else:
        edge_weight = torch.ones(data.edge_index.size(1)).to(torch.float).unsqueeze(-1)

    edge_index = data.edge_index
    data = T.ToSparseTensor()(data)

    if args.dataset == "ogbl-citation2":
        source_nodes = split_edge['valid']['source_node']
        target_nodes = split_edge['valid']['target_node']
        val_edge_index = torch.stack((source_nodes, target_nodes))
    elif args.dataset == "ogbl-ppa":
        # Read in subset
        subset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "ppa_subset")
        val_pos_ix = torch.load(os.path.join(subset_dir, "valid_samples_index.pt"))
        val_edge_index = split_edge['valid']['edge'][val_pos_ix, :].t()
    else:
        val_edge_index = split_edge['valid']['edge'].t()

    val_edge_index = to_undirected(val_edge_index)
    full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)

    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
    val_edge_weight = torch.cat([edge_weight, val_edge_weight], 0).view(-1)

    data.train_valid_edge_index = full_edge_index
    data.train_valid_adj = SparseTensor.from_edge_index(full_edge_index, val_edge_weight, [data.num_nodes, data.num_nodes])

    if args.dataset == "ogbl-citation2":
        val_pos = torch.stack((split_edge['valid']['source_node'], split_edge['valid']['target_node'])).t()
        test_pos = torch.stack((split_edge['test']['source_node'], split_edge['test']['target_node'])).t()
    elif args.dataset == "ogbl-ppa":
        # Read in subset
        subset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "ppa_subset")
        val_pos_ix = torch.load(os.path.join(subset_dir, "valid_samples_index.pt"))
        test_pos_ix = torch.load(os.path.join(subset_dir, "test_samples_index.pt"))

        val_pos = split_edge['valid']['edge'][val_pos_ix, :]
        test_pos = split_edge['test']['edge'][test_pos_ix, :]
    else:
        val_pos  = split_edge['valid']['edge']
        test_pos = split_edge['test']['edge']

    data_obj = {
        "dataset": args.dataset,
        "x": data.x,
        "adj_t": data.adj_t,
        "edge_index": edge_index,
        "num_nodes": data.num_nodes,
        "train_valid_adj": data.train_valid_adj,
        "train_valid_edge_index": data.train_valid_edge_index,
        "valid_pos": val_pos,
        "test_pos": test_pos
    }    

    return data_obj


def get_data_planetoid(data_name):
    """
    Get data for cora/citeseer/pubmed
    """
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []

    for split in ['train', 'test', 'valid']:
        path = dir_path + '/dataset' + '/{}/{}_pos.txt'.format(data_name, split)

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': train_pos.append((sub, obj))
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test':  test_pos.append((sub, obj))
    
    num_nodes = len(node_set)

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

    valid_pos = torch.tensor(valid_pos)
    test_pos =  torch.tensor(test_pos)
    
    val_edge_index = valid_pos.t()
    val_edge_index = to_undirected(val_edge_index)
    full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
    train_valid_adj = SparseTensor.from_edge_index(full_edge_index, torch.ones(full_edge_index.size(1)), [num_nodes, num_nodes])      

    feature_embeddings = torch.load(dir_path + '/dataset' + '/{}/{}'.format(data_name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {
        "dataset": data_name,
        "adj_t": adj,
        "edge_index": edge_index,
        "train_valid_adj": train_valid_adj,
        "train_valid_edge_index": full_edge_index,
        "num_nodes": num_nodes,
        "x": feature_embeddings,
        "valid_pos": valid_pos,
        "test_pos": test_pos
    }

    return data