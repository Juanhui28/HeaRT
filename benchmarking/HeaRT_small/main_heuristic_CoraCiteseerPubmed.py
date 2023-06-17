import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from collections import Counter

import sys
sys.path.append("..") 

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from get_heuristic import *
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from utils import*


dir_path = get_data_dir()


def read_data(data_name, dir_path, filename):
    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:

       
        path = dir_path + '/{}/{}_pos.txt'.format(data_name, split)

     
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                

            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    with open(f'{dir_path}/{data_name}/valid_{filename}', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/test_{filename}', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)

    

    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    train_pos_tensor =  torch.transpose(torch.tensor(train_pos), 1, 0)

    valid_pos =  torch.transpose(torch.tensor(valid_pos), 1, 0)
    test_pos =  torch.transpose(torch.tensor(test_pos), 1, 0)
    
    valid_neg =  torch.tensor(valid_neg)
    test_neg =  torch.tensor(test_neg)

    valid_neg = torch.permute(valid_neg, (2, 0, 1))
    valid_neg = valid_neg.view(2,-1)

    test_neg = torch.permute(test_neg, (2, 0, 1))
    test_neg = test_neg.view(2,-1)
    

    return  A, train_pos_tensor, valid_pos, test_pos, valid_neg, test_neg, train_pos


def get_prediction(A, full_A, use_heuristic, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge):

    # if 'katz' in use_heuristic:
    #     pos_val_pred = eval(use_heuristic)( A, pos_val_edge)
    #     neg_val_pred = eval(use_heuristic)( A, neg_val_edge)

    #     pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge, beta, path_len, remove)
    #     neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge, beta, path_len, remove)

    # if use_heuristic == 'shortest_path':
    #     pos_val_pred = eval(use_heuristic)( A, pos_val_edge, remove)
    #     neg_val_pred = eval(use_heuristic)( A, neg_val_edge, remove)

    #     pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge, remove)
    #     neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge, remove)

    pos_val_pred = eval(use_heuristic)(A, pos_val_edge)
    neg_val_pred = eval(use_heuristic)(A, neg_val_edge)

    pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge)
    neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge)

    return pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred


def get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
   
    result = {}
    k_list = [1, 3, 10, 100]
   
    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in [1,3,10, 100]:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--use_heuristic', type=str, default='katz_apro')
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, default=get_data_dir())
    parser.add_argument('--filename', type=str, default='samples.npy')

    parser.add_argument('--beta', type=float, default='0.005')

    args = parser.parse_args()

    # dataset = Planetoid('.', 'cora')

    A, train_pos, valid_pos, test_pos, valid_neg, test_neg, train_pos_list  = read_data(args.data_name, args.input_dir, args.filename)

    train_edge = torch.transpose(torch.tensor(train_pos_list), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    print('val pos, val neg, test pos, test neg', valid_pos.size(), valid_neg.size(), test_pos.size(), test_neg.size())

    node_num = A.shape[0]

    if args.use_valedges_as_input:
        print('use validation!!!')
        val_edge_index = valid_pos
        val_edge_index = to_undirected(val_edge_index)

        edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1)], dtype=int)

        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        


        full_A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                        shape=(node_num, node_num)) 
        print('nonzero values: ', full_A.nnz)
    else:
        
        full_A = A
        print('no validation!!!')
        print('nonzero values: ', full_A.nnz)


    use_heuristic = args.use_heuristic

    pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred = get_prediction(A, full_A, use_heuristic, valid_pos, valid_neg, test_pos, test_neg)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    # Counter(pos_test_pred.numpy())

   
    neg_val_pred = neg_val_pred.view( pos_val_pred.size(0), -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)

    print('pred val pos, val neg, test pos, test neg', pos_val_pred.size(), neg_val_pred.size(), pos_test_pred.size(), neg_test_pred.size())

    results = get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    print('heurisitic: ', args.use_heuristic)  

    for key, result in results.items():
        train_hits, valid_hits, test_hits = result
        print(key)
        print( f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')




   
if __name__ == "__main__":
    main()