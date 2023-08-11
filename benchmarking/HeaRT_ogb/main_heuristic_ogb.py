import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from collections import Counter

import sys, os
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
from evalutors import evaluate_hits, evaluate_auc, evaluate_mrr


def get_prediction(A, full_A, use_heuristic, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge, args):

    beta, path_len = args.beta, args.path_len
    remove = args.remove

    if 'katz' in use_heuristic:
        pos_val_pred = eval(use_heuristic)( A, pos_val_edge, beta, path_len, remove)
        neg_val_pred = eval(use_heuristic)( A, neg_val_edge, beta, path_len, remove)

        pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge, beta, path_len, remove)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge, beta, path_len, remove)

    elif use_heuristic == 'shortest_path':
        pos_val_pred = eval(use_heuristic)( A, pos_val_edge, remove)
        neg_val_pred = eval(use_heuristic)( A, neg_val_edge, remove)

        pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge, remove)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge, remove)

    else:  
        print('evaluate pos vlaid: ')
        pos_val_pred = eval(use_heuristic)(A, pos_val_edge)
        print('evaluate neg vlaid: ')
        neg_val_pred = eval(use_heuristic)(A, neg_val_edge)


        print('evaluate pos test: ')
        pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge)
        print('evaluate neg test: ')

        neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge)

    return pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred

def get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_val_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
   
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--use_heuristic', type=str, default='CN')

    
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--use_mrr', action='store_true', default=False)

    ####### katz
    parser.add_argument('--beta', type=float, default=0.005)
    parser.add_argument('--path_len', type=int, default=3)

    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--remove', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    

    args = parser.parse_args()
    print(args)

    # dataset = Planetoid('.', 'cora')

    dataset = PygLinkPropPredDataset(name=args.data_name, root=os.path.join(get_root_dir(), "dataset", args.data_name))
    
    data = dataset[0]
    
    use_heuristic = args.use_heuristic
    split_edge = dataset.get_edge_split()
    node_num = data.num_nodes
    edge_index = data.edge_index

    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            # edge_weight = data.edge_weight.to(torch.float)
            edge_weight = data.edge_weight.view(-1).to(torch.float)
           
        else:
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    else:
        
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                       shape=(node_num, node_num))
    
    if args.data_name == 'ogbl-collab':
        pos_train_edge = split_edge['train']['edge']

        pos_valid_edge = split_edge['valid']['edge']
        
        pos_test_edge = split_edge['test']['edge']
    
        
        with open(f'{args.input_dir}/{args.data_name}/heart_valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{args.data_name}/heart_test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
    
    elif args.data_name == 'ogbl-ppa':
        pos_train_edge = split_edge['train']['edge']
        
        subset_dir = f'{args.input_dir}/{args.data_name}'
        val_pos_ix = torch.load(os.path.join(subset_dir, "valid_samples_index.pt"))
        test_pos_ix = torch.load(os.path.join(subset_dir, "test_samples_index.pt"))

        pos_valid_edge = split_edge['valid']['edge'][val_pos_ix, :]
        pos_test_edge = split_edge['test']['edge'][test_pos_ix, :]

       
        with open(f'{args.input_dir}/{args.data_name}/heart_valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{args.data_name}/heart_test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
    


    
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(1), target_edge.unsqueeze(1)], dim=-1)

        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        # neg_valid_edge = split_edge['valid']['target_node_neg'] 

        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        # neg_test_edge = split_edge['test']['target_node_neg']

        
        with open(f'{args.input_dir}/{args.data_name}/heart_valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{args.data_name}/heart_test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
        
        idx = torch.tensor([1,0])
        edge_index = torch.cat([edge_index, edge_index[idx]], dim=1)
        edge_weight = torch.ones(edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                       shape=(node_num, node_num))

    if args.use_valedges_as_input:
        print('use validation!!!')
        val_edge_index = pos_valid_edge.transpose(1, 0)
        val_edge_index = to_undirected(val_edge_index)

        edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1)], dtype=int)

        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        


        full_A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                        shape=(node_num, node_num)) 
        
    else:
        print('no validation!!!')

        full_A = A
    
    print('A:', A.nnz)
    print('full_A:', full_A.nnz)

    pos_valid_edge = pos_valid_edge.transpose(1, 0)
    pos_test_edge = pos_test_edge.transpose(1, 0)
    neg_valid_edge = torch.permute(neg_valid_edge, (2, 0, 1))
    neg_test_edge = torch.permute(neg_test_edge, (2, 0 ,1) )

    neg_valid_edge = neg_valid_edge.view(2, -1)
    neg_test_edge = neg_test_edge.view(2, -1)
   
    print('edge size', pos_valid_edge.size(), neg_valid_edge.size(), pos_test_edge.size(), neg_test_edge.size())
    pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred = get_prediction(A, full_A, use_heuristic, pos_valid_edge, neg_valid_edge, pos_test_edge, neg_test_edge, args)

    state = {
        'pos_val': pos_val_pred,
        'neg_val': neg_val_pred,
        'pos_test': pos_test_pred,
        'neg_test': neg_test_pred
    }
   

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    # Counter(pos_test_pred.numpy())

    neg_val_pred = neg_val_pred.view( pos_val_pred.size(0), -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)

    print('pred size', pos_val_pred.size(), neg_val_pred.size(), pos_test_pred.size(), neg_test_pred.size())

    results = get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    print('heurisitic: ', args.use_heuristic)    
  
    for key, result in results.items():
        train_hits, valid_hits, test_hits = result
        print(key)
        print( f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
       
    
    save_path = args.output_dir + '/beta'+ str(args.beta) + '_pathlen'+ str(args.path_len) + '_' + 'save_score'
    # torch.save(state, save_path)

if __name__ == "__main__":
 

    main()