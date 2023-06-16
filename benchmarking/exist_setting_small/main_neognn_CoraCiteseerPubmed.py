
import sys
sys.path.append("..") 

import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
from scoring import mlp_score
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

from baseline_models.neognn import NeoGNN
from torch_scatter import scatter_add

dir_path = '.'
log_print		= get_logger('testrun', 'log', '../config/')
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

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
          

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]


    feature_embeddings = torch.load(dir_path+'/dataset' + '/{}/{}'.format(data_name, 'gnn_feature'))
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

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 3, 10, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in [1,3,10, 100]:
    #     result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

   
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

        

def train(model, score_func, A, train_pos, x,  optimizer, batch_size,args):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    count = 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()


        num_nodes = x.size(0)

        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
    
        train_edge_mask = train_pos[mask].transpose(1,0)

        # train_edge_mask = to_undirected(train_edge_mask)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        # edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
       
        ###################
        # print(adj)

        edge = train_pos[perm].t()
        pos_out, pos_out_struct, pos_out_feat, _ = model(edge,  adj, A, x, num_nodes, score_func, only_feature=args.only_use_feature)
        
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                            device=edge.device)
        neg_out, neg_out_struct, neg_out_feat, _ = model(edge, adj, A, x, num_nodes, score_func, only_feature=args.only_use_feature)



        if pos_out_struct != None:
            pos_loss = -torch.log(pos_out_struct + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
            loss1 = pos_loss + neg_loss
        else:
            loss1 = 0


        if pos_out_feat != None:
            pos_loss = -torch.log(pos_out_feat + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out_feat + 1e-15).mean()
            loss2 = pos_loss + neg_loss
        else:
            loss2 = 0

        if pos_out != None:
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            loss3 = pos_loss + neg_loss
        else:
            loss3 = 0


        # if count == 0:
        #     print('loss1 loss2 loss3: ', loss1.item(), loss2.item(), loss3.item())
            
        loss = loss1 + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out_feat.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        count += 1
        # if count % 50 == 0:
        #     break

    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size, alpha, model, A_):

  

    preds = []
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        gnn_scores = score_func(h[edge[0]], h[edge[1]]).squeeze().cpu()
        src, dst = input_data[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())  
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        preds += [cur_scores]
    pred_all = torch.cat(preds, dim=0)

    return pred_all


@torch.no_grad()
def test(model, score_func, data, A, num_nodes, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    

    h = model.forward_feature(data['x'], data['adj'])

    edge_weight = torch.from_numpy(A.data).to(h.device)
    edge_weight = model.f_edge(edge_weight.unsqueeze(-1))

    row, col = A.nonzero()
    edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(h.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg =  model.f_node(deg).squeeze()

    deg = deg.cpu().numpy()
    A_ = A.multiply(deg).tocsr()

    alpha = torch.softmax(model.alpha, dim=0).cpu()
    # print(alpha) 
   
    pos_train_pred = test_edge(score_func, data['train_val'], h, batch_size,  alpha, model, A_)

    neg_valid_pred = test_edge(score_func, data['valid_neg'], h, batch_size, alpha, model, A_)

    pos_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size, alpha, model, A_)

    pos_test_pred = test_edge(score_func, data['test_pos'], h, batch_size, alpha, model, A_)

    neg_test_pred = test_edge(score_func, data['test_neg'], h, batch_size, alpha, model, A_)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    return result

def data_to_device(data, device):

    data['adj'] = data['adj'].to(device)
    data['train_pos'] = data['train_pos'].to(device)
    data['train_val'] = data['train_val'].to(device)
    data['valid_pos'] = data['valid_pos'].to(device)
    data['valid_neg'] = data['valid_neg'].to(device)
    data['test_pos'] = data['test_pos'].to(device)
    data['test_neg'] = data['test_neg'].to(device)
    data['x'] = data['x'].to(device)

    return data

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='NeoGNN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)

    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ######neo-gnn
    parser.add_argument('--f_edge_dim', type=int, default=8) 
    parser.add_argument('--f_node_dim', type=int, default=128) 
    parser.add_argument('--g_phi_dim', type=int, default=128) 
    parser.add_argument('--only_use_feature',	action='store_true',   default=False,   	help='whether only use the feature')
    parser.add_argument('--beta', type=float, default=0.1)
	



    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.neg_mode)

    input_channel = data['x'].size(1)
    node_num = data['x'].size(0)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout,args).to(device)
       
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    
    data = data_to_device(data, device)

    # x = data['x'].to(device)
    # train_pos = data['train_pos'].to(x.device)

    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs)
       
    }

    edge_index = data['train_pos'].cpu().t()
    edge_weight = torch.ones(edge_index.size(1), dtype=float)
    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                       shape=(node_num, node_num))
    A2 = A * A
    A = A + args.beta*A2
    


    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        
        save_path = args.output_dir+'/best_run_'+str(run)


        model.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, score_func, A, data['train_pos'], data['x'], optimizer, args.batch_size, args)
            # print(model.convs[0].att_src[0][0][:10])
           
            if epoch % args.eval_steps == 0:
                results_rank = test(model, score_func, data, A, node_num,  evaluator_hit, evaluator_mrr, args.batch_size)

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
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.save:
                       
                        save_model(model, save_path, emb=None)
                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean


            
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        



    
    
    print(best_metric_valid_str +' ' +best_auc_valid_str)

    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()
   