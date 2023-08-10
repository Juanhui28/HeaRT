"""
utility functions and global variables
"""

import os
from distutils.util import strtobool
from math import inf

import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DEFAULT_DIC = {'sample_size': None, 'dataset_name': 'Cora', 'num_hops': 2, 'max_dist': 10, 'max_nodes_per_hop': 10,
               'data_appendix': None, 'val_pct': 0.1, 'test_pct': 0.2, 'train_sample': 1, 'dynamic_train': True,
               'dynamic_val': True, 'model': 'hashing', 'sign_k': 2,
               'dynamic_test': True, 'node_label': 'drnl', 'ratio_per_hop': 1, 'use_feature': True, 'dropout': 0,
               'label_dropout': 0, 'feature_dropout': 0,
               'add_normed_features': False, 'use_RA': False, 'hidden_channels': 32, 'load_features': True,
               'load_hashes': True, 'use_zero_one': True, 'wandb': False, 'batch_size': 32, 'num_workers': 1,
               'cache_subgraph_features': False, 'eval_batch_size': 1000, 'num_negs': 1}


def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)


def get_num_samples(sample_arg, dataset_len):
    """
    convert a sample arg that can be a number of % into a number of samples
    :param sample_arg: float interpreted as % if < 1 or count if >= 1
    :param dataset_len: the number of data points before sampling
    :return:
    """
    if sample_arg < 1:
        samples = int(sample_arg * dataset_len)
    else:
        samples = int(min(sample_arg, dataset_len))
    return samples


def select_embedding(args, num_nodes, device):
    """
    select a node embedding. Used by SEAL models (the E in SEAL is for Embedding)
    and needed for ogb-ddi where there are no node features
    :param args: Namespace of cmd args
    :param num_nodes: Int number of nodes to produce embeddings for
    :param device: cpu or cuda
    :return: Torch.nn.Embedding [n_nodes, args.hidden_channels]
    """
    if args.train_node_embedding:
        emb = torch.nn.Embedding(num_nodes, args.hidden_channels).to(device)
    elif args.pretrained_node_embedding:
        weight = torch.load(args.pretrained_node_embedding)
        emb = torch.nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad = False
    else:
        emb = None
    return emb




def str2bool(x):
    """
    hack to allow wandb to tune boolean cmd args
    :param x: str of bool
    :return: bool
    """
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')
