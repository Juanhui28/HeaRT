"""
utility functions and global variables
"""

import os
from distutils.util import strtobool
from math import inf

import torch
import numpy as np
import json, logging, sys
import math
import logging.config 
import random
from torch_scatter import scatter_add

def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_print_logger():

    config_dir = 'config/'
    name = 'log'
    config_dict = json.load(open( config_dir + 'log_config.json'))
    # config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def get_metric_logger(args):
    
    if args.data_name == 'cora' or args.data_name == 'citeseer' or args.data_name == 'pubmed':
        logger = {
                'Hits@1': Logger(args.runs),
                'Hits@3': Logger(args.runs),
                'Hits@10': Logger(args.runs),
                'Hits@100': Logger(args.runs),
                'MRR': Logger(args.runs)
            }
        metric = 'MRR'
    
    elif 'citation2' in args.data_name:
        logger = {
                'Hits@20': Logger(args.runs),
                'Hits@50': Logger(args.runs),
                'Hits@100': Logger(args.runs),
                'MRR': Logger(args.runs)
            }
        metric = 'MRR'
    
    else:
        logger = {
                'Hits@20': Logger(args.runs),
                'Hits@50': Logger(args.runs),
                'Hits@100': Logger(args.runs),
            }
        
        if 'collab' in args.data_name:  metric = 'Hits@50'
        elif 'ddi' in args.data_name:  metric = 'Hits@20'
        elif 'ppa' in args.data_name:  metric = 'Hits@100'

    return logger, metric


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')

            r = best_result[:, 0].float()
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 2].float()
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]


            return best_valid, best_valid_mean, mean_list, var_list

def set_device(args):
   
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    return device
    




def save_emb(score_emb, save_path):

    if len(score_emb) == 6:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x1, x2= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x1,
        'node_emb_with_valid_edges': x2

        }
        
    elif len(score_emb) == 5:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x
        }
    elif len(score_emb) == 4:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, = score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        }
   
    torch.save(state, save_path)


def print_gating_inf(gating_inf, start_str):

    gating_score = torch.cat(gating_inf[0])
    gating_ind = torch.cat(gating_inf[1])
    mess = start_str+' load nmuber: '
    for i in range(8):
        mess += f'{round((gating_ind == i).sum().item(),4)}, '
    print(mess)
    mess = start_str + ' ave score: '
    for i in range(8):
        mess += f'{round(gating_score[gating_ind == i].mean().item(),4)}, '
    print(mess)
    print('---')


def get_neognn_A(args, A, model, device, num_nodes):
    edge_weight = torch.from_numpy(A.data).to(device)
    if args.data_name not in ['ogbl-ppa', 'ogbl-citation2']:
        edge_weight_A = torch.from_numpy(A.data).to(torch.float64)
    else:
        edge_weight_A = torch.from_numpy(A.data)

    edge_weight = model.f_edge(edge_weight.unsqueeze(-1))

    row, col = A.nonzero()
    edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg =  model.f_node(deg).squeeze()

    deg = deg.cpu().numpy()
    A_ = A.multiply(deg).tocsr()

    return A_