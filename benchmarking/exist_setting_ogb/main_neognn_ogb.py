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

from baseline_models.neognn import NeoGNN
from torch_scatter import scatter_add
import os
from torch_geometric.utils import negative_sampling

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

        

def train_ppa(model, score_func, A, train_pos, data, optimizer, batch_size, gnn_batch_size, only_use_feature):
    model.train()
    score_func.train()

    total_loss = total_examples = 0
    
    # if emb == None: 
    #     x = data.x
    #     emb_update = 0
    # else: 
    #     x = emb.weight
    #     emb_update = 1

    count = 0
    
    for perm, perm_large in zip(DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True),  DataLoader(range(train_pos.size(0)), gnn_batch_size,
                           shuffle=True)):
        optimizer.zero_grad()
        num_nodes = data.x.size(0)
       
        adj = data.adj_t 

        edge = train_pos[perm].t()
        pos_out, pos_out_struct, pos_out_feat, pos_out_struct_raw = model(edge,  adj, A, data.x, num_nodes, score_func, only_feature=only_use_feature)

        edge_large = train_pos[perm_large].t()
        edge_large = torch.randint(0, num_nodes, edge_large.size(), dtype=torch.long, device=edge.device)
        with torch.no_grad():
            h = model.forward_feature(data.x, data.adj_t)
            neg_out_gnn_large = score_func(h[edge_large[0]], h[edge_large[1]])
            neg_large_loss = -torch.log(1 - (torch.sigmoid(neg_out_gnn_large)) + 1e-15)
            edge = edge_large[:,torch.topk(neg_large_loss.squeeze(), batch_size)[1]]
        
        neg_out, neg_out_struct, neg_out_feat, neg_out_struct_raw = model(edge, adj, A, data.x, num_nodes, score_func, only_feature=only_use_feature)


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
            
        loss = loss1 + loss2 + loss3 +  1e-3 * (torch.abs(pos_out_struct_raw).mean() + torch.abs(neg_out_struct_raw).mean())
        loss.backward()
       
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out_feat.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        count += 1
        if count % 50 == 0:
            break

    return total_loss / total_examples

def train(model, score_func, A, train_pos, data, emb, optimizer, batch_size, args, pos_train_weight, data_name, remove_edge_aggre):
            
            
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    # train_pos = split_edge['train']['edge'].to(data.x.device)
    
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1

    count = 0
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
        # print(adj)

        edge = train_pos[perm].t()
        pos_out, pos_out_struct, pos_out_feat,  pos_out_struct_raw = model(edge,  adj, A, x, num_nodes, score_func, only_feature=args.only_use_feature)
    

        if data_name == 'ogbl-ddi':
            # print('ddiddiddididdddddd')
            row, col, _ = adj.coo()
            edge_index = torch.stack([col, row], dim=0)
            edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                    num_neg_samples=perm.size(0), method='dense')
        else:
            # Just do some trivial random sampling.
            edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                                device=edge.device)
        
        neg_out, neg_out_struct, neg_out_feat, neg_out_struct_raw = model(edge, adj, A, x, num_nodes, score_func, only_feature=args.only_use_feature)



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
        # + 1e-3 * (torch.abs(pos_out_struct_raw).mean() + torch.abs(neg_out_struct_raw).mean())
        loss.backward()
        if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out_feat.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        count += 1
        if count % 50 == 0:
            break

    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size, alpha, model, A_, mrr_mode=False, negative_data=None):

  

    preds = []
    if mrr_mode:
        source = input_data.t()[0]
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = negative_data.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm].cpu(), target_neg[perm].cpu()
            gnn_scores = score_func(h[src], h[dst_neg]).squeeze().cpu()

            cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst_neg]), 1)).to(h.device)
            cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())  
            cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
            preds += [cur_scores]

        pred_all = torch.cat(preds, dim=0).view(-1, 1000)

    else:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            gnn_scores = score_func(h[edge[0]], h[edge[1]]).squeeze().cpu()
            # print(input_data[perm].t().size())
            src, dst = input_data[perm].t().cpu()
            
            cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
            cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())  
            cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
            preds += [cur_scores]
        pred_all = torch.cat(preds, dim=0)

    return pred_all

@torch.no_grad()
def test_citation2(model, score_func, data, evaluation_edges,A,  emb,  num_nodes, evaluator_hit, evaluator_mrr, batch_size, data_name, use_valedges_as_input):
    model.eval()
    score_func.eval()

    model.eval()
    score_func.eval()

    if emb == None: x = data.x
    else: x = emb.weight
    
    h = model.forward_feature(x, data.adj_t)

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

    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges

    
   
    # print(h[0][:10])
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)

    neg_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size,  alpha, model, A_, mrr_mode=True, negative_data=neg_valid_edge)

    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size,  alpha, model, A_)

    pos_test_pred = test_edge(score_func, pos_test_edge, h, batch_size,  alpha, model, A_)

    neg_test_pred = test_edge(score_func, pos_test_edge, h, batch_size, alpha, model, A_, mrr_mode=True, negative_data=neg_test_edge)

    pos_train_pred = test_edge(score_func, train_val_edge, h, batch_size,  alpha, model, A_)
        
    pos_valid_pred = pos_valid_pred.view(-1)
    pos_test_pred =pos_test_pred.view(-1)
    pos_train_pred = pos_valid_pred.view(-1)
    
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]

    return result, score_emb

def get_A(A, model, device, num_nodes):
    edge_weight = torch.from_numpy(A.data).to(device)
    edge_weight = model.f_edge(edge_weight.unsqueeze(-1))

    row, col = A.nonzero()
    edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg =  model.f_node(deg).squeeze()

    deg = deg.cpu().numpy()
    A_ = A.multiply(deg).tocsr()

    return A_


@torch.no_grad()
def test(model, score_func, data, evaluation_edges,A,  emb,  num_nodes, evaluator_hit, evaluator_mrr, batch_size, data_name, use_valedges_as_input,full_A):
    model.eval()
    score_func.eval()

    if emb == None: x = data.x
    else: x = emb.weight
    
    h = model.forward_feature(x, data.adj_t)
    device = h.device
    A_ = get_A(A, model, device, data.num_nodes)

    alpha = torch.softmax(model.alpha, dim=0).cpu()


    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges

    
    
    # print(h[0][:10])
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)

    pos_train_pred = test_edge(score_func, train_val_edge, h,batch_size,  alpha, model, A_)

    neg_valid_pred = test_edge(score_func, neg_valid_edge, h, batch_size,  alpha, model, A_)

    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size,  alpha, model, A_)

    if use_valedges_as_input:
        print('use vali!!!!!!!!!!!!!!!!!')
        h = model.forward_feature(x, data.full_adj_t)
       
        A_ = get_A(full_A, model, device, data.num_nodes)



    pos_test_pred = test_edge(score_func, pos_test_edge, h, batch_size,  alpha, model, A_)

    neg_test_pred = test_edge(score_func, neg_test_edge, h, batch_size,  alpha, model, A_)



    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, data_name)
    
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]

    return result, score_emb



# def main(count, lr, l2, dropout):
def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='NeoGNN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='Hits@50')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)
    
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
    parser.add_argument('--test_batch_size', type=int, default=1024 * 64)
	
  
    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('use_val_edge:', args.use_valedges_as_input)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    dataset = PygLinkPropPredDataset(name=args.data_name, root=os.path.join(get_root_dir(), "dataset", args.data_name))
    
    data = dataset[0]

    edge_index = data.edge_index
    emb = None
    node_num = data.num_nodes
    split_edge = dataset.get_edge_split()

    if hasattr(data, 'x'):
        if data.x != None:
            data.x = data.x.to(torch.float)
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
    full_A = None

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)

        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
    
        data.full_adj_t = A
        data.full_edge_index = full_edge_index

        edge_weight = torch.ones(data.full_edge_index.size(1), dtype=float)
        full_A = ssp.csr_matrix((edge_weight, (full_edge_index[0], full_edge_index[1])), 
                        shape=(data.num_nodes, data.num_nodes))
        
        A2 = full_A * full_A
        full_A = full_A + args.beta*A2
        

        print(data.full_adj_t)
        print(data.adj_t)

    else:
        data.full_adj_t = data.adj_t

    if args.data_name == 'ogbl-citation2': 
        data.adj_t = data.adj_t.to_symmetric()
        edge_index = to_undirected(edge_index)
        print('citation2 adj:', data.adj_t)
        print('citation2 edge_index:', edge_index.size())



        
        
    data = data.to(device)
   
   
    model = NeoGNN(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout,args).to(device)
       
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    
    # train_pos = data['train_pos'].to(x.device)

    # eval_metric = args.metric
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

    if args.data_name != 'ogbl-citation2':
        pos_train_edge = split_edge['train']['edge']

        pos_valid_edge = split_edge['valid']['edge']
        neg_valid_edge = split_edge['valid']['edge_neg']
        pos_test_edge = split_edge['test']['edge']
        neg_test_edge = split_edge['test']['edge_neg']
    
    else:
        
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(1), target_edge.unsqueeze(1)], dim=-1)

        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        neg_valid_edge = split_edge['valid']['target_node_neg'] 

        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        neg_test_edge = split_edge['test']['target_node_neg']


   

    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]

    pos_train_edge = pos_train_edge.to(device)


    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]

    if args.data_name =='ogbl-ppa' or args.data_name =='ogbl-citation2':
        edge_weight = torch.ones(edge_index.size(1))
    else:
        edge_weight = torch.ones(edge_index.size(1), dtype=float)
        
    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                       shape=(data.num_nodes, data.num_nodes))
    
    if args.data_name =='ogbl-collab':
        A2 = A * A
        A = A + args.beta*A2
        degree = torch.from_numpy(A.sum(axis=0)).squeeze()

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
        for epoch in range(1, 1 + args.epochs):
            
                
            if args.data_name == 'ogbl-ppa':
                print('data: ppa')
                loss = train_ppa(model, score_func, A, pos_train_edge, data, optimizer, args.batch_size, args.test_batch_size, args.only_use_feature)
            else:
                loss = train(model, score_func, A, pos_train_edge, data, emb, optimizer, args.batch_size, args, train_edge_weight, args.data_name, args.remove_edge_aggre)
       
            if epoch % args.eval_steps == 0:
                # results_rank = test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, args.test_batch_size, args.data_name, args.use_valedges_as_input)
                if args.data_name == 'ogbl-citation2':
                    results_rank, score_emb = test_citation2(model, score_func, data, evaluation_edges,A,  emb,  node_num, evaluator_hit, evaluator_mrr, args.test_batch_size, args.data_name, args.use_valedges_as_input)
                
                else:
                    results_rank, score_emb = test(model, score_func, data, evaluation_edges,A,  emb,  node_num, evaluator_hit, evaluator_mrr, args.test_batch_size, args.data_name, args.use_valedges_as_input, full_A)

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

                    
                       
                        # save_model(model, save_path, emb=emb)
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
        print(best_metric_valid_str +' ' +best_auc_valid_str)

    # return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":

    main()


    
    