

import sys
sys.path.append("..") 

import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from gnn_model import *

from scoring import mlp_score
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_auc, evaluate_mrr
# from evaluate_mrr_hit import evaluate_mrr
from torch_geometric.utils import negative_sampling
import os
import copy as cp

from baseline_models.seal_dataset import SEALDataset, SEALDynamicDataset
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

from baseline_models.seal_utils import *
import time
from gnn_model import *
from torch.nn import BCEWithLogitsLoss
from utils import *

dir_path = get_root_dir()
log_print		= get_logger('testrun', 'log', get_config_dir())



def get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100]
    result = {}

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
    for K in [20, 50, 100]:
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

def train(model, train_loader,  optimizer, device, args, emb, train_dataset):
    model.train()
   
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
    # for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)

@torch.no_grad()
def test_multiple_models(models, val_loader, test_loader, device, args, emb, evaluator_hit, evaluator_mrr):
    for m in models:
        # m = m.to(device)
        m.eval()


    y_pred, y_true = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, m in enumerate(models):
            logits = m(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred[i].append(logits.view(-1).cpu())
            y_true[i].append(data.y.view(-1).cpu().to(torch.float))
    test_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
    test_true = [torch.cat(y_true[i]) for i in range(len(models))]
    pos_test_pred = [test_pred[i][test_true[i]==1] for i in range(len(models))]
    neg_test_pred = [test_pred[i][test_true[i]==0] for i in range(len(models))]

    pos_val_pred = pos_test_pred
    neg_val_pred = neg_test_pred

    if val_loader != None:
        y_pred, y_true = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
        for data in tqdm(val_loader, ncols=70):
            data = data.to(device)
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            for i, m in enumerate(models):
                logits = m(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
                y_pred[i].append(logits.view(-1).cpu())
                y_true[i].append(data.y.view(-1).cpu().to(torch.float))
        val_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
        val_true = [torch.cat(y_true[i]) for i in range(len(models))]
        pos_val_pred = [val_pred[i][val_true[i]==1]  for i in range(len(models))]
        neg_val_pred = [val_pred[i][val_true[i]==0]  for i in range(len(models))]

    
    print('val pos, val neg, test pos, test neg:',  pos_val_pred[0].size(), neg_val_pred[0].size(), pos_test_pred[0].size(), neg_test_pred[0].size())
    
   
    # result = get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_val_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    
    
    Results = []
    for i in range(len(models)):
        result = get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_val_pred[0], pos_val_pred[0], neg_val_pred[0], pos_test_pred[0], neg_test_pred[0])
    
        Results.append(result)

       
    return Results
   
   

   

@torch.no_grad()
def test(model, val_loader, test_loader,  device, args,emb, evaluator_hit, evaluator_mrr, data_name):
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(val_loader, ncols=70):
    # for data in val_loader:
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader, ncols=70):
    # for data in test_loader:
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]
    
    print('val pos, val neg, test pos, test neg:',  pos_val_pred.size(), neg_val_pred.size(), pos_test_pred.size(), neg_test_pred.size())

    if data_name == 'ogbl-citation2':
        result = get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_val_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    
    else:

        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    
    score_emb = [pos_val_pred.cpu(),neg_val_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb
    

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='DGCNN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=30,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='Hits@50')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ##### seal
    parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
    parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
    parser.add_argument('--train_node_embedding', action='store_true', 
                    help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--test_percent', type=float, default=100)
    parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
    parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--use_feature', action='store_true', 
                    help="whether to use raw node features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
    parser.add_argument('--test_multiple_models', action='store_true', 
                    help="test multiple models together")
    parser.add_argument('--test_seed', type=int,default=0)

    parser.add_argument('--model_path', type=str, default='output_test')
    parser.add_argument('--val_full', action='store_true', 
                    help="full validation")
    
   

    args = parser.parse_args()

    print('use_val_edge:', args.use_valedges_as_input)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.data_name, root=os.path.join(get_root_dir(), "dataset", args.data_name))
    data = dataset[0]


    if args.save_appendix == '':
        args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
    if args.data_appendix == '':
        args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)

    edge_index = data.edge_index
    emb = None
    node_num = data.num_nodes
    split_edge = dataset.get_edge_split()


    
    # data = T.ToSparseTensor()(data)
    if args.data_name == 'ogbl-citation2': directed=True
    else: directed = False

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        if not directed:
            val_edge_index = to_undirected(val_edge_index)
        
        full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)
 
        data.edge_index = full_edge_index
        data.edge_weight = edge_weight


    if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
        args.num_workers = 0

    path = 'dataset/'+ str(args.data_name)+ '_seal{}'.format(args.data_appendix)
    use_coalesce = True if args.data_name == 'ogbl-collab' else False
    

    dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
    train_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.train_percent, 
        split='train', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    ) 
    

    dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
    val_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.val_percent, 
        split='valid', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    )
    dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
    test_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.test_percent, 
        split='test', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    )

    max_z = 1000  # set a large max_z so that every z has embeddings to look up

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers)

    if args.data_name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ogbl-ddi':
        eval_metric = 'Hits@20'

    elif args.data_name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    
    elif args.data_name =='ogbl-citation2':
        eval_metric = 'MRR'
    
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
    if args.train_node_embedding:
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    else:
        emb = None
    
    if args.test_multiple_models:
        model_paths = [] 
        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
                      train_dataset, args.dynamic_train, use_feature=args.use_feature, 
                      node_embedding=emb).to(device)
        # for run in range(args.test_seed):
        run=0
        # file = 'output/citation2/seal/SEAL_model_'+str(args.lr)+'_'+str(args.test_seed)
        file = args.model_path+'/SEAL_model_'+str(args.test_seed)

        model_paths.append(file)
            # enter all your pretrained .pth model paths here
        models = []
        for path in model_paths:
            m = cp.deepcopy(model)
            m.load_state_dict(torch.load(path, map_location='cpu'))
            models.append(m)

        if args.val_full == False:
            val_loader = None
        Results = test_multiple_models(models, val_loader, test_loader, device, args, emb, evaluator_hit, evaluator_mrr)

        for i, path in enumerate(model_paths):
            print(path)
           
            results = Results[i]
            for key, result in results.items():
                loggers[key].add_result(run, result)
            for key, result in results.items():
                valid_res, valid_res, test_res = result
                
                to_print = (f'Run: {run + 1:02d}, ' +
                            f'Valid: {100 * valid_res:.2f}%, ' +
                            f'Test: {100 * test_res:.2f}%')
                print(key)
                print(to_print)
               
        for key in loggers.keys():
            if len(loggers[key].results[0]) > 0:
                print(key)
                _,  _, _, _ = loggers[key].print_statistics()
        return


    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run

        print('seed: ', seed)
        init_seed(seed)

        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
                      train_dataset, args.dynamic_train, use_feature=args.use_feature, 
                      node_embedding=emb).to(device)
        parameters = list(model.parameters())
        if args.train_node_embedding:
            torch.nn.init.xavier_uniform_(emb.weight)
            parameters += list(emb.parameters())
        
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.l2)
        if args.gnn_model == 'DGCNN':
            print(f'SortPooling k is set to {model.k}')

        best_valid = 0
        kill_cnt = 0
        best_valid_auc = best_test_auc = 2
        best_auc_valid_str = 2
        
        for epoch in range(1, 1 + args.epochs):
           
            loss = train(model, train_loader,  optimizer, device, args, emb, train_dataset)
            
            if epoch % args.eval_steps == 0:
                print('testing')
                results_rank , score_emb= test(model, val_loader, test_loader,  device, args,emb, evaluator_hit, evaluator_mrr, args.data_name)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                r = torch.tensor(loggers[eval_metric].results[run])
                best_valid_current = round(r[:, 1].max().item(),4)
                best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                if len(loggers['AUC'].results[run]) > 0:
                    r = torch.tensor(loggers['AUC'].results[run])
                    best_valid_auc = round(r[:, 1].max().item(), 4)
                    best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)

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
                    
                    print(eval_metric)
                    log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                   f'best test: {100*best_test:.2f}%')

                    print('AUC')
                    log_print.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                   f'best test: {100*best_test_auc:.2f}%')

                    print('---')
                
                
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0


                    if args.save:

                        save_emb(score_emb, save_path)
                        torch.save(model.state_dict(), args.output_dir+'/SEAL_model_'+str(seed))
                        torch.save(optimizer.state_dict(), args.output_dir+'/SEAL_opti_'+str(seed))
                
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
    
    
        print(best_metric_valid_str +' ' +best_auc_valid_str)


if __name__ == "__main__":

    main()
