
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
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_auc, evaluate_mrr
# from evaluate_mrr_hit import evaluate_mrr
from torch_geometric.utils import negative_sampling
import os

dir_path = '..'
log_print		= get_logger('testrun', 'log', dir_path+'/config/')


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
   
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result



        

def train_use_hard_negative(model, score_func, train_pos, data, emb, optimizer, batch_size, pos_train_weight, remove_edge_aggre, gnn_batch_size):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1


    
    for perm, perm_large in zip(DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True),  DataLoader(range(train_pos.size(0)), gnn_batch_size,
                           shuffle=True)):
        
        optimizer.zero_grad()


        num_nodes = x.size(0)

        ######################### remove loss edges from the aggregation

        
        if remove_edge_aggre:

            mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
            mask[perm] = 0
        
            train_edge_mask = train_pos[mask].transpose(1,0)
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)

            if pos_train_weight != None:
                edge_weight_mask = pos_train_weight[mask]
                edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
            

            else:
                edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)

        else:
           
            adj = data.adj_t 
             
        ###################
        # print(adj)

        h = model(x, adj)

        edge = train_pos[perm].t()

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        
        edge_large = train_pos[perm_large].t()
        edge_large = torch.randint(0, num_nodes, edge_large.size(), dtype=torch.long, device=edge.device)
        with torch.no_grad():
           
            neg_out_gnn_large = score_func(h[edge_large[0]], h[edge_large[1]])
            neg_large_loss = -torch.log(1 - (torch.sigmoid(neg_out_gnn_large)) + 1e-15)
            edge = edge_large[:,torch.topk(neg_large_loss.squeeze(), batch_size)[1]]
        

        
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples



def train(model, score_func, split_edge, train_pos, data, emb, optimizer, batch_size, pos_train_weight, data_name, remove_edge_aggre):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1


    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()


        num_nodes = x.size(0)

        ######################### remove loss edges from the aggregation

        
        if remove_edge_aggre:

            mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
            mask[perm] = 0
        
            train_edge_mask = train_pos[mask].transpose(1,0)
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)

            if pos_train_weight != None:
                edge_weight_mask = pos_train_weight[mask]
                edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
            

            else:
                edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)

        else:
           
            adj = data.adj_t 
             
        ###################

        h = model(x, adj)

        edge = train_pos[perm].t()

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                                device=h.device)
            
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size,  negative_data=None):

    
    # preds = []

    # if negative_data != None:
    #     source = input_data.t()[0]
    #     source = source.view(-1, 1).repeat(1, 1000).view(-1)
    #     target_neg = negative_data.view(-1)

    #     for perm in DataLoader(range(source.size(0)), batch_size):
    #         src, dst_neg = source[perm], target_neg[perm]
    #         preds += [score_func(h[src], h[dst_neg]).squeeze().cpu()]
    #     pred_all = torch.cat(preds, dim=0).view(-1, 1000)

    # else:

    #     for perm  in DataLoader(range(input_data.size(0)), batch_size):
    #         edge = input_data[perm].t()
        
    #         preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
            
    #     pred_all = torch.cat(preds, dim=0)


    pos_preds = []
    neg_preds = []

    if negative_data is not None:
        
        for perm in DataLoader(range(input_data.size(0)),  batch_size):
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))

            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()

            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
            
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds, neg_preds

@torch.no_grad()
def test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, batch_size, use_valedges_as_input):
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges

    if emb == None: x = data.x
    else: x = emb.weight
    
    h = model(x, data.adj_t.to(x.device))
    x1 = h
    x2 = torch.tensor(1)
    # print(h[0][:10])
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)

   
    
    pos_valid_pred, neg_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size, negative_data=neg_valid_edge)

    if use_valedges_as_input:
        print('use_val_in_edge')
        h = model(x, data.full_adj_t.to(x.device))
        x2 = h

    pos_test_pred, neg_test_pred = test_edge(score_func, pos_test_edge, h, batch_size, negative_data=neg_test_edge)
    
    # print(' test_pos test_neg', pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
   
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = pos_valid_pred

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)
   
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]

    return result, score_emb



# def main(count, lr, l2, dropout):
def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-ppa')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=20,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default='dataset')
    parser.add_argument('--filename', type=str, default='samples_agg-min_norm-1.npy')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ##### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    parser.add_argument('--use_hard_negative', default=False, action='store_true')

    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--device', type=int, default=0)
    
    ##### debug
    # parser.add_argument('--gnn_model', type=str, default='GAT')
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--hidden_channels', type=int, default=256)
    # parser.add_argument('--device', type=int, default=5)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--num_layers', type=int, default=0)
    # parser.add_argument('--num_layers_predictor', type=int, default=3)
    # parser.add_argument('--hidden_channels', type=int, default=256)
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--gnn_model', type=str, default='MF')



    args = parser.parse_args()
    # args.lr = lr
    # args.l2 = l2
    # args.dropout = dropout

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('use_val_edge:', args.use_valedges_as_input)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print('use_hard_negative: ',args.use_hard_negative)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    dataset = PygLinkPropPredDataset(name=args.data_name)
    
    data = dataset[0]

    edge_index = data.edge_index
    emb = None
    node_num = data.num_nodes
    split_edge = dataset.get_edge_split()

    if hasattr(data, 'x'):
        if data.x != None:
            x = data.x
            data.x = data.x.to(torch.float)

            if args.cat_n2v_feat:
                print('cat n2v embedding!!')
                n2v_emb = torch.load('dataset/'+args.data_name+'-n2v-embedding.pt')
                data.x = torch.cat((data.x, n2v_emb), dim=-1)
            
            data.x = data.x.to(device)

            input_channel = data.x.size(1)

        else:

            emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
            input_channel = args.hidden_channels

    else:
        emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
        input_channel = args.hidden_channels
    
    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            edge_weight = data.edge_weight.to(torch.float)
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            train_edge_weight = split_edge['train']['weight'].to(device)
            train_edge_weight = train_edge_weight.to(torch.float)
        else:
            train_edge_weight = None

    else:
        train_edge_weight = None

    
    data = T.ToSparseTensor()(data)

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)

        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)
    
    if args.data_name == 'ogbl-citation2': 
        data.adj_t = data.adj_t.to_symmetric()
        if args.gnn_model == 'GCN':
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t

            
    data = data.to(device)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout, mlp_layer=args.gin_mlp_layer, head=args.gat_head, node_num=node_num, cat_node_feat_mf=args.cat_node_feat_mf,  data_name=args.data_name).to(device)

    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
   
   
    
    # train_pos = data['train_pos'].to(x.device)

    # eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)

    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs)
      
    }

    if args.data_name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ogbl-ddi':
        eval_metric = 'Hits@20'

    elif args.data_name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    
    elif args.data_name =='ogbl-citation2':
        eval_metric = 'MRR'

    if args.data_name == 'ogbl-collab':
        pos_train_edge = split_edge['train']['edge']

        pos_valid_edge = split_edge['valid']['edge']
        
        pos_test_edge = split_edge['test']['edge']
    
        read_data_name = args.data_name.replace('-', '_')
        with open(f'{args.input_dir}/{read_data_name}/valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{read_data_name}/test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
    
    elif args.data_name == 'ogbl-ppa':
        pos_train_edge = split_edge['train']['edge']
        read_data_name = args.data_name.replace('-', '_')
        subset_dir = f'{args.input_dir}/{read_data_name}'
        val_pos_ix = torch.load(os.path.join(subset_dir, "valid_samples_index.pt"))
        test_pos_ix = torch.load(os.path.join(subset_dir, "test_samples_index.pt"))

        pos_valid_edge = split_edge['valid']['edge'][val_pos_ix, :]
        pos_test_edge = split_edge['test']['edge'][test_pos_ix, :]

       
        with open(f'{args.input_dir}/{read_data_name}/valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{read_data_name}/test_{args.filename}', "rb") as f:
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

        read_data_name = args.data_name.replace('-', '_')
        with open(f'{args.input_dir}/{read_data_name}/valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{read_data_name}/test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)


    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]

    pos_train_edge = pos_train_edge.to(device)


    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    print('train val val_neg test test_neg: ', pos_train_edge.size(), pos_valid_edge.size(), neg_valid_edge.size(), pos_test_edge.size(), neg_test_edge.size())
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
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)
        
        # save_valid = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'valid_output'+str(run)
        # save_valid = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'valid_output'+str(run)
        

        if emb != None:
            torch.nn.init.xavier_uniform_(emb.weight)

        model.reset_parameters()
        score_func.reset_parameters()

        if emb != None:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=args.lr, weight_decay=args.l2)

        else:
            optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        best_test = 0
        
        for epoch in range(1, 1 + args.epochs):
            if args.use_hard_negative:
                loss = train_use_hard_negative(model, score_func, pos_train_edge, data, emb, optimizer, args.batch_size, train_edge_weight, args.remove_edge_aggre, args.test_batch_size)
            else:
                loss = train(model, score_func, split_edge, pos_train_edge, data, emb, optimizer, args.batch_size, train_edge_weight, args.data_name, args.remove_edge_aggre)
           
            
            if epoch % args.eval_steps == 0:
              
                results_rank, score_emb= test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, args.test_batch_size, args.use_valedges_as_input)

                
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
                
               
                print('---')
                
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save: 
                        torch.save(model.state_dict(), save_path+'_model')
                        torch.save(optimizer.state_dict(),save_path+'_op')
                        torch.save(score_func.state_dict(), save_path+'_predictor')
                        torch.save(emb,save_path+'_emb')
                        save_emb(score_emb, save_path)

                
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

    # return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":

    main()


    