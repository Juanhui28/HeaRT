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

from utils import *
from get_heuristic import *
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc


dir_path = get_root_dir()


def read_data(data_name, neg_mode):
    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:

        if neg_mode == 'equal':
            path = dir_path+'/dataset' + '/{}/{}_pos.txt'.format(data_name, split)

        elif neg_mode == 'all':
            path = dir_path+'/dataset' + '/{}/allneg/{}_pos.txt'.format(data_name, split)

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

    for split in ['test', 'valid']:

        if neg_mode == 'equal':
            path = dir_path+'/dataset' + '/{}/{}_neg.txt'.format(data_name, split)

        elif neg_mode == 'all':
            path = dir_path+'/dataset' + '/{}/allneg/{}_neg.txt'.format(data_name, split)

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            
            if split == 'valid': 
                valid_neg.append((sub, obj))
               
            if split == 'test': 
                test_neg.append((sub, obj))

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))


    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    train_pos_tensor =  torch.transpose(torch.tensor(train_pos), 1, 0)

    valid_pos =  torch.transpose(torch.tensor(valid_pos), 1, 0)
    valid_neg =  torch.transpose(torch.tensor(valid_neg), 1, 0)

    test_pos =  torch.transpose(torch.tensor(test_pos), 1, 0)
    test_neg =  torch.transpose(torch.tensor(test_neg), 1, 0)



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

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    k_list  = [1, 3, 10, 100]
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)
    
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    
    result = {}

    result_hit = {}
    for K in [1, 3, 10, 100]:
        result[f'Hits@{K}'] = (result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])

    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))
    
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1))
    
    result_mrr = {}
    result['MRR'] = (result_mrr_val['MRR'], result_mrr_test['MRR'])
    result['mrr_hit1']  = (result_mrr_val['mrr_hit1'], result_mrr_test['mrr_hit1'])
    result['mrr_hit3']  = (result_mrr_val['mrr_hit3'], result_mrr_test['mrr_hit3'])
    result['mrr_hit10']  = (result_mrr_val['mrr_hit10'], result_mrr_test['mrr_hit10'])
    result['mrr_hit100']  = (result_mrr_val['mrr_hit100'], result_mrr_test['mrr_hit100'])
   

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    result_auc = {}
    result['AUC'] = (result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_val['AP'], result_auc_test['AP'])


    return result

        


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--use_heuristic', type=str, default='katz_apro')
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)

    parser.add_argument('--beta', type=float, default='0.005')

    args = parser.parse_args()

    # dataset = Planetoid('.', 'cora')

    A, train_pos, valid_pos, test_pos, valid_neg, test_neg, train_pos_list  = read_data(args.data_name, args.neg_mode)

    train_edge = torch.transpose(torch.tensor(train_pos_list), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

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

    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)



    print('heurisitic: ', args.use_heuristic)    

    print('valid/test mrr of ' + args.data_name + ' is: ', result['MRR'][0], result['MRR'][1])

    print('hit 1, 3, 10, 100 of ' + args.data_name + ' is: ', result['Hits@1'][1], result['Hits@3'][1], result['Hits@10'][1], result['Hits@100'][1])


    print('AUC and AP of ' + args.data_name + ' is: ', result['AUC'][1], result['AP'][1] )

    print('hit 1, 3, 10 from mrr of ' + args.data_name + ' is: ', result['mrr_hit1'][1], result['mrr_hit3'][1], result['mrr_hit10'][1],  result['mrr_hit100'][1])



if __name__ == "__main__":
    main()