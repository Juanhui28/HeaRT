import os
import sys
import math
import pprint
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append("..") 
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

import scipy.sparse as ssp


from baseline_models.nbfnet import tasks, util, datasets
from baseline_models.nbfnet.util import detect_variables, literal_eval

from torch_sparse import SparseTensor

from utils import get_logger, save_emb, init_seed, Logger
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
import easydict
import argparse
import numpy as np

from torch_geometric.data import DataLoader


separator = ">" * 30
line = "-" * 30

dir_path = '.'
log_print		= get_logger('testrun', 'log', '../config/')

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

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
          

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    valid_neg = valid_neg.view(-1, 2)
    test_neg = test_neg.view(-1, 2)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]


    feature_embeddings = torch.load(dir_path+'/{}/{}'.format(data_name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {}
    data['adj'] = adj
    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg

    data['x'] = feature_embeddings

    return data

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
   
    result = {}
    k_list = [1, 3, 10, 100]
   
    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in [1,3,10, 100]:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    
    return result


def train_and_validate(cfg, model, train_data, valid_data, test_data, device, run,  eval_log, working_dir=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    # step = math.ceil(cfg.train.num_epoch / 10)
    step = 1
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    
    save_path = cfg.output_dir+'/lr'+str(cfg.optimizer.lr) + '_drop' + str(cfg.model.dropout) + '_l2' + str(cfg.optimizer.weight_decay) +   '_seed'+str(cfg.seed)+ '_best_run_'+str(run)
    
    kill_cnt = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            # if util.get_rank() == 0:
                # logger.warning(separator)
                # logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in train_loader:
                # new_batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                #                                 strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
                target = torch.ones_like(pred)
                
                pos_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean()
                
                neg_edge = torch.randint(0, train_data.num_nodes, (2, batch.size(0)), dtype=torch.long,
                             device=device)
                neg_edge = torch.transpose(neg_edge,1,0)
                neg_edge_with_type = torch.cat((neg_edge, batch[:,2].unsqueeze(1)), dim=-1)
                pred_neg = parallel_model(train_data, neg_edge_with_type)
                target_neg = torch.zeros_like(pred_neg)
                neg_loss = F.binary_cross_entropy_with_logits(pred_neg, target_neg, reduction="none").mean()
                
                loss = pos_loss + neg_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


        epoch = min(cfg.train.num_epoch, i + step)
        
        # if rank == 0:
        #     logger.warning(separator)
            # logger.warning("Evaluate on valid and test")
        all_result, score_emb = test(cfg, model, valid_data, test_data, evaluator_hit, evaluator_mrr)
        
        for key, result in all_result.items():
            eval_log[key].add_result(run, result)

        r = torch.tensor(eval_log[cfg.train.eval_metric].results[run])
        best_valid_current = round(r[:, 1].max().item(),4)
        best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

      
        
        for key, result in all_result.items():
            
            print(key)
            
            train_hits, valid_hits, test_hits = result
            log_print.info(
                f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_hits:.2f}%, '
                    f'Valid: {100 * valid_hits:.2f}%, '
                    f'Test: {100 * test_hits:.2f}%')

        print(cfg.train.eval_metric)
        log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                        f'best test: {100*best_test:.2f}%')


        print('---')

        result = all_result[cfg.train.eval_metric][1]
        if result > best_result:
            best_result = result
            best_epoch = epoch
            kill_cnt = 0
            if cfg.save :
                save_emb(score_emb, save_path)
        else:
            kill_cnt += 1
            if kill_cnt > cfg.train.kill_cnt: 
                print("Early Stopping!!")
                break
            

    for key in eval_log.keys():
        if len(eval_log[key].results[0]) > 0:
            print(key)
            eval_log[key].print_statistics(run)

   

    return eval_log


@torch.no_grad()
def test_edge(cfg, model, target_edge, test_data, world_size, rank, mode=None):

    if mode != None:
        edge_type =  test_data.target_edge_type.unsqueeze(0)
        neg_num =  int(target_edge.shape[1] / edge_type.shape[1])
        edge_type = edge_type.repeat(1, neg_num)
        test_triplets = torch.cat([target_edge, edge_type]).t()


    else:
        test_triplets = torch.cat([target_edge, test_data.target_edge_type.unsqueeze(0)]).t()

    test_loader = DataLoader(test_triplets, batch_size=cfg.test_bs, shuffle=False)

    model.eval()

    preds = []
    for batch in test_loader:
        pre = model(test_data, batch)
        pre = torch.sigmoid(pre)

        preds += [pre.cpu()]
        
    pred_all = torch.cat(preds, dim=0)

    return pred_all

@torch.no_grad()
def test(cfg, model, val_data, test_data,  evaluator_hit, evaluator_mrr):
    world_size = util.get_world_size()
    rank = util.get_rank()

    

    pos_valid_pred = test_edge(cfg, model, val_data.target_edge_index, val_data, world_size, rank)

    neg_valid_pred = test_edge(cfg, model, val_data.target_neg, val_data, world_size, rank, mode='neg')

    pos_test_pred = test_edge(cfg, model, test_data.target_edge_index, test_data, world_size, rank)

    neg_test_pred = test_edge(cfg, model, test_data.target_neg, test_data, world_size, rank, mode='neg')

    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)

    neg_valid_pred = neg_valid_pred.view(pos_valid_pred.size(0), -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)

    print(' valid_pos valid_neg test_pos test_neg', pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", default = '../baseline_models/nbfnet/data_config/cora.yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=999)
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--lr', type=float, default=5.0e-3)
    parser.add_argument('--input_dim', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.1) 
    parser.add_argument('--hidden_dims',  nargs='+', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default='dataset')
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=64)

    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_bs', type=int, default=1024)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--runs', type=int, default=10)


    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    
    ######
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    ####
    # vars = dict()
    # vars['gpus'] = '[2]'
    #####
    return args, vars

if __name__ == "__main__":
    
    args, vars = parse_args()


    data = read_data(args.data_name, args.input_dir, args.filename)
    cfg = util.load_config(args.config, context=vars)
    print(args)
    
    init_seed(args.seed)
  
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    is_inductive = cfg.dataset["class"].startswith("Ind")
    train_data,valid_data, test_data,dataset = util.build_dataset(cfg, data)
    
    device = util.get_device(cfg)

    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    
    eval_log = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs)
    }


    for run in range(args.runs):
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        init_seed(seed)

        cfg = util.load_config(args.config, context=vars)
        cfg.model.num_relation = dataset.num_relations
        cfg.model.hidden_dims = args.hidden_dims
        cfg.model.input_dim = args.input_dim
        cfg.model.dropout = args.dropout
        cfg.optimizer.lr = args.lr
        cfg.optimizer.weight_decay = args.weight_decay
        cfg.seed = seed
        cfg.output_dir = args.output_dir
        cfg.train.batch_size = args.batch_size
        cfg.train.num_epoch = args.num_epoch
        cfg.save = args.save
        cfg.test_bs = args.test_bs
        # working_dir = util.create_working_directory(cfg)
        print(cfg)

        model = util.build_model(cfg)
       
        model = model.to(device)

        eval_log = train_and_validate(cfg, model, train_data, valid_data, test_data, device, run, eval_log)
    


    for key in eval_log.keys():
        if len(eval_log[key].results[0]) > 0:
            print(key)
            
            best_metric,  best_valid_mean, mean_list, var_list = eval_log[key].print_statistics()

            if key == cfg.train.eval_metric:
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean

                
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean

    if args.runs == 1:
        
        r = torch.tensor(eval_log[cfg.train.eval_metric].results[0])
        best_valid_current = round(r[:, 1].max().item(),4)
        best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

     

        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_current) + ' ' + str(best_test))
    
    else:

        best_auc_valid_str = best_metric_valid_str
        print(best_metric_valid_str +' ' +best_auc_valid_str)
    