import time
import warnings
from math import inf
import sys

sys.path.insert(0, '..')

from utils import *
import numpy as np
import torch
from ogb.linkproppred import Evaluator
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_networkx, to_undirected
from baseline_models.BUDDY.data import get_loaders
from baseline_models.BUDDY.utils import select_embedding, select_model, get_num_samples, get_loss, get_split_samples, str2bool, get_ogb_data
from torch.utils.data import DataLoader
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
import argparse
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from baseline_models.BUDDY.utils import filter_by_year

log_print		= get_logger('testrun', 'log', '../config/')


def get_data(args):
    dataset_name = args.data_name
    
    directed = False

    dataset = PygLinkPropPredDataset(name=dataset_name)
    if dataset_name == 'ogbl-ddi':
        dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
        dataset.data.edge_weight = torch.ones(dataset.data.edge_index.size(1), dtype=int)
    
    if dataset_name.startswith('ogbl-citation'):
        eval_metric = 'mrr'
        directed = True
    
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    if dataset_name == 'ogbl-collab' and args.year > 0:  # filter out training edges before args.year
        data, split_edge = filter_by_year(data, split_edge, args.year)
    splits = get_ogb_data(data, split_edge, dataset_name, args.num_negs)
    
    return dataset, splits, directed


def get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100]
    result = {}
    
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])

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

def  train(model, optimizer, train_loader, args, device, emb=None):

    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]

    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(loader):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(device)

        if args.use_struct_feature:
            sf_indices = sample_indices[indices]  # need the original link indices as these correspond to sf
            subgraph_features = data.subgraph_features[sf_indices].to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links].to(device)
        degrees = data.degrees[curr_links].to(device)
        if args.use_RA:
            ra_indices = sample_indices[indices]
            RA = data.RA[ra_indices].to(device)
        else:
            RA = None
        
        optimizer.zero_grad()
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
    

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_edge(model, loader, device, args, split=None):

    n_samples = get_split_samples(split, args, len(loader.dataset))
    preds = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.to(device))
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    for batch_count, indices in enumerate(loader):
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(device)
        if args.use_struct_feature:
            subgraph_features = data.subgraph_features[indices].to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links].to(device)
        degrees = data.degrees[curr_links].to(device)
        if args.use_RA:
            RA = data.RA[indices].to(device)
        else:
            RA = None
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break

    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    return pos_pred, neg_pred



@torch.no_grad()
def test(model, evaluator_hit, evaluator_mrr, train_loader, val_loader, test_loader, args, device):
   
    model.eval()

        
    if  'citation2' in args.data_name:
        pos_valid_pred, neg_valid_pred = test_edge(model, val_loader, device, args, split='val')
    
        pos_test_pred, neg_test_pred  = test_edge(model, test_loader, device, args, split='test')

        pos_valid_pred = pos_valid_pred.view(-1)
        pos_test_pred =pos_test_pred.view(-1)
        pos_train_pred = pos_valid_pred.view(-1)

        
        print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
        
        result = get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    else:
        pos_train_pred, neg_train_pred = test_edge(model, train_loader, device, args, split='train')
    
        pos_valid_pred, neg_valid_pred = test_edge(model, val_loader, device, args, split='val')
        
        pos_test_pred, neg_test_pred  = test_edge(model, test_loader, device, args, split='test')

        pos_valid_pred = pos_valid_pred.view(-1)
        pos_test_pred =pos_test_pred.view(-1)
        pos_train_pred = pos_train_pred.view(-1)


        pos_train_pred = torch.flatten(pos_train_pred)
        pos_valid_pred = torch.flatten(pos_valid_pred)
        neg_test_pred =  torch.flatten(neg_test_pred)

        print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
        
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb



def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    
    ##gnn setting
    
    parser.add_argument('--hidden_channels', type=int, default=256)
    

    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=20,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    ##
    parser.add_argument('--model', type=str, default='BUDDY')
    parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    parser.add_argument('--floor_sf', type=str2bool, default=0,
                        help='the subgraph features represent counts, so should not be negative. If --floor_sf the min is set to 0')
    parser.add_argument('--minhash_num_perm', type=int, default=128, help='the number of minhash perms')
    parser.add_argument('--hll_p', type=int, default=8, help='the hyperloglog p parameter')
    parser.add_argument('--use_zero_one', type=str2bool,
                        help="whether to use the counts of (0,1) and (1,0) neighbors")
    parser.add_argument('--load_features', action='store_true', help='load node features from disk')
    parser.add_argument('--load_hashes', action='store_true', help='load hashes from disk')
    parser.add_argument('--cache_subgraph_features', action='store_true',
                        help='write / read subgraph features from disk')
    parser.add_argument('--use_feature', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--use_struct_feature', type=str2bool, default=True,
                        help="whether to use structural graph features as GNN input")
    parser.add_argument('--loss', default='bce', type=str, help='bce or auc')

    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=1024*64,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
    parser.add_argument('--sign_dropout', type=float, default=0.5)
    ###
    # parser.add_argument('--device', type=int, default=1)
    
    args = parser.parse_args()
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset, splits, directed = get_data(args)
    train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
    

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
        'mrr_hit20':  Logger(args.runs),
        'mrr_hit50':  Logger(args.runs),
        'mrr_hit100':  Logger(args.runs),
    }

    if args.data_name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ogbl-ddi':
        eval_metric = 'Hits@20'

    elif args.data_name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    
    elif args.data_name =='ogbl-citation2':
        eval_metric = 'MRR'
    
    best_valid_auc = best_test_auc = 2
    best_auc_valid_str = 2

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_l2'+ str(args.l2)  +'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        emb = select_embedding(args, dataset.data.num_nodes, device)
        model, optimizer = select_model(args, dataset, emb, device)

        best_valid = 0
        kill_cnt = 0
        best_test = 0

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, optimizer, train_loader, args, device)

            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(model, evaluator_hit, evaluator_mrr, train_loader, val_loader, test_loader, args, device)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)


                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

                    r = torch.tensor(loggers[eval_metric].results[run])
                    best_valid_current = round(r[:, 1].max().item(),4)
                    best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                    print(eval_metric)
                    log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                   f'best test: {100*best_test:.2f}%')
                    
                    if len(loggers['AUC'].results[run]) > 0:
                        r = torch.tensor(loggers['AUC'].results[run])
                        best_valid_auc = round(r[:, 1].max().item(), 4)
                        best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)
                        
                        print('AUC')
                        log_print.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                    f'best test: {100*best_test_auc:.2f}%')
                    
                    print('---')
                
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save: save_emb(score_emb, save_path)

                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            if len(loggers[key].results[0]) > 0:
                print(key)
                loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():
        if len(loggers[key].results[0]) > 0:
            print(key)
            
            best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

            if key == eval_metric:
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean


                
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean

            result_all_run[key] = [mean_list, var_list]
            



        
    if args.runs == 1:
        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))
    
    else:
        print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))

   

if __name__ == "__main__":
    main()
