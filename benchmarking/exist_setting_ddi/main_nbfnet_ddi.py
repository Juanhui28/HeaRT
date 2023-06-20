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


from baseline_models.nbfnet import tasks, util

from torch_sparse import SparseTensor

from utils import get_logger, save_emb, init_seed, Logger
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
import easydict

separator = ">" * 30
line = "-" * 30



def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,data_name):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [20, 50, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])

    if data_name =='ogbl-citation2':

        result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
        result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
        result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
        
        # result_mrr = {}
        result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
        for K in [20, 50, 100]:
            result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

   
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    
    return result


def train_and_validate(cfg, model, train_data, valid_data, test_data, device, run, eval_log, working_dir=None):
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

    step = math.ceil(cfg.train.num_epoch / 10)
    
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    # save_path = cfg.output_dir+'/lr'+str(cfg.optimizer.lr) + '_drop' + str(cfg.model.dropout) +  '_numlayer' + str(len(cfg.model.hidden_dims)) + '_'+ 'best_run_'+str(run)
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
        
        
        all_result, score_emb = test(cfg, model, valid_data, test_data, evaluator_hit, evaluator_mrr)
        
        for key, result in all_result.items():
            eval_log[key].add_result(run, result)

        r = torch.tensor(eval_log[cfg.train.eval_metric].results[run])
        best_valid_current = round(r[:, 1].max().item(),4)
        best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

        r = torch.tensor(eval_log['AUC'].results[run])
        best_valid_auc = round(r[:, 1].max().item(), 4)
        best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)
        
        for key, result in all_result.items():
            
            print(key)
            
            train_hits, valid_hits, test_hits = result
            logger.warning(
                f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_hits:.2f}%, '
                    f'Valid: {100 * valid_hits:.2f}%, '
                    f'Test: {100 * test_hits:.2f}%')

        print(cfg.train.eval_metric)
        logger.warning(f'best valid: {100*best_valid_current:.2f}%, '
                        f'best test: {100*best_test:.2f}%')

        print('AUC')
        logger.warning(f'best valid: {100*best_valid_auc:.2f}%, '
                        f'best test: {100*best_test_auc:.2f}%')


        print('---')

        result = all_result[cfg.train.eval_metric][1]
        if result > best_result:
            best_result = result
            best_epoch = epoch
            kill_cnt = 0
            # save_emb(score_emb, save_path)
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
def test_edge(cfg, model, target_edge, target_edge_type, test_data, world_size, rank):

    test_triplets = torch.cat([target_edge, target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

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

    

    pos_valid_pred = test_edge(cfg, model, val_data.target_edge_index, val_data.target_edge_type, val_data, world_size, rank)

    neg_valid_pred = test_edge(cfg, model, val_data.target_neg, val_data.target_neg_type, val_data, world_size, rank)

    pos_test_pred = test_edge(cfg, model, test_data.target_edge_index, test_data.target_edge_type, test_data, world_size, rank)

    neg_test_pred = test_edge(cfg, model, test_data.target_neg, test_data.target_neg_type, test_data, world_size, rank)

    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    print('train valid_pos valid_neg test_pos test_neg', pos_valid_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb



if __name__ == "__main__":
    
    args, vars = util.parse_args()
    # print('111111: ', args)
    # print('222222: ', vars)
  
    cfg = util.load_config(args.config, context=vars)
    print(args)
    # working_dir = util.create_working_directory(cfg)

    # print('333333: ', cfg)

    init_seed(args.seed)
    # torch.manual_seed(args.seed + util.get_rank())
    

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    is_inductive = cfg.dataset["class"].startswith("Ind")
    train_data,valid_data, test_data,dataset = util.build_dataset(cfg)

    device = util.get_device(cfg)

    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    
    eval_log = {
        'Hits@20': Logger(cfg.train.runs),
        'Hits@50': Logger(cfg.train.runs),
        'Hits@100': Logger(cfg.train.runs),
        'MRR': Logger(cfg.train.runs),
        'AUC':Logger(cfg.train.runs),
        'AP':Logger(cfg.train.runs)
 
    }
    

    for run in range(cfg.train.runs):
        
        if cfg.train.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        init_seed(seed)

        cfg = util.load_config(args.config, context=vars)
        cfg.model.num_relation = dataset.num_relations

        cfg.model.hidden_dims = args.hidden_dims
        cfg.model.dropout = args.dropout
        cfg.optimizer.lr = args.lr
        cfg.output_dir = args.output_dir
        # working_dir = util.create_working_directory(cfg)

        model = util.build_model(cfg)
       
        model = model.to(device)

        eval_log = train_and_validate(cfg, model, train_data, valid_data, test_data, device, seed, eval_log)
    
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

    if cfg.train.runs == 1:
        
        r = torch.tensor(eval_log[cfg.train.eval_metric].results[0])
        best_valid_current = round(r[:, 1].max().item(),4)
        best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

        r = torch.tensor(eval_log['AUC'].results[0])
        best_valid_auc = round(r[:, 1].max().item(), 4)
        best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)

        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))
    
    else:

    
        print(best_metric_valid_str +' ' +best_auc_valid_str)
    
    