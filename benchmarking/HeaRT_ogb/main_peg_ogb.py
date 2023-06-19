import sys
sys.path.append("..") 
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import networkx as nx
from utils import Logger, get_logger, save_emb, init_seed, get_root_dir, get_config_dir
from baseline_models.PEG.PEGLayer_collab import PEGconv
import scipy.sparse as sp
import tensorflow
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected


from baseline_models.PEG.Graph_embedding import DeepWalk
from evalutors import evaluate_hits, evaluate_auc, evaluate_mrr
import time
import os
import pickle
import numpy as np

import dgl

import numpy as np
import networkx as nx
import random
import math
from sklearn.preprocessing import normalize

dir_path = get_root_dir()
log_print		= get_logger('testrun', 'log', get_config_dir())

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

#modified from: https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/collab
class PEG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(PEG, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(PEGconv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                PEGconv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(PEGconv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, embeddings):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, embeddings)
        x = self.convs[-1](x, adj_t, embeddings)
        return x





class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.output = torch.nn.Linear(2,1)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, pos_i, pos_j):
        x = x_i * x_j
        pos_encode = ((pos_i - pos_j)**2).sum(dim=-1, keepdim=True)
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        out = self.output(torch.cat([x, pos_encode], 1))
        

        return torch.sigmoid(out)


def train(model, predictor, data, embeddings, split_edge, optimizer, batch_size):

    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(data.x, edge_index, embeddings)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples




@torch.no_grad()
def test(model, predictor, data, embeddings, split_edge, evaluator_hit,evaluator_mrr, batch_size, use_valedges_as_input):

    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)
    print('edge index 1:', edge_index.size())

    model.eval()
    predictor.eval()

    h = model(data.x, edge_index, embeddings)
    #h = model(data.x, edge_index)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

   
    pos_valid_preds = []
    neg_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]]).squeeze().cpu()]

        # neg_edges = torch.permute(neg_valid_edge[perm], (2, 0, 1))
        neg_edges = torch.transpose(neg_valid_edge[perm], 2, 0)
        neg_edges = torch.transpose(neg_edges, 2, 1)
        neg_edges = neg_edges.view(2,-1)
        neg_valid_preds += [predictor(h[neg_edges[0]], h[neg_edges[1]], embeddings[neg_edges[0]], embeddings[neg_edges[1]]).squeeze().cpu()]


    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    if use_valedges_as_input:

        # row, col, _ = data.adj_t.coo()
        # edge_index = torch.stack([col, row], dim=0)

        row, col, _ = data.full_adj_t.coo()
        edge_index = torch.stack([col, row], dim=0)
        print('edge index 2:', edge_index.size())
        # edge
        h = model(data.x, edge_index, embeddings)

       

    pos_test_preds = []
    neg_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]]).squeeze().cpu()]

        # neg_edges = torch.permute(neg_test_edge[perm], (2, 0, 1))
        neg_edges = torch.transpose(neg_test_edge[perm], 2, 0)
        neg_edges = torch.transpose(neg_edges, 2, 1)
        neg_edges = neg_edges.view(2,-1)
        neg_test_preds += [predictor(h[neg_edges[0]], h[neg_edges[1]], embeddings[neg_edges[0]], embeddings[neg_edges[1]]).squeeze().cpu()]

    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = pos_valid_pred

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)

    neg_valid_pred = neg_valid_pred.view(pos_valid_pred.size(0), -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    out = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return out

def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--PE_method', type=str, default='DW')
    parser.add_argument('--PE_dim', type=int, default=128)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--no_pe', action='store_true', default=False, help = 'whether to use pe')
    parser.add_argument('--seed', type=int, default=999)

    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)


    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=20,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=4096)

    # parser.add_argument('--use_valedges_as_input', action='store_true', default=True)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if not os.path.exists('dataset/peg'):
        os.makedirs('dataset/peg')
    dataset = PygLinkPropPredDataset(name=args.data_name, root='dataset/peg/')
    data = dataset[0]
    edge_index = data.edge_index

    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
   
    

    split_edge = dataset.get_edge_split()

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)
        edge_weight = data.edge_weight.to(torch.float)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)

        val_edge_weight = torch.ones([val_edge_index.size(1)], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)

        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        
        data.full_adj_t = A
        data.full_edge_index = full_edge_index


        print('full adj:',data.full_adj_t)
        
    data = T.ToSparseTensor()(data)
    data_save = args.data_name.replace('-', '_')
    if args.PE_method == 'DW':
        
        if os.path.exists('dataset/'+data_save+'/dw_emb_collab'):
            embeddings = torch.load('dataset/'+data_save+'/dw_emb_collab', map_location=torch.device('cpu'))

        else:

            G = nx.from_edgelist(np.array(dataset[0].edge_index).T)
            model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
            model_emb.train(embed_size = args.PE_dim)# train model
            emb = model_emb.get_embeddings()# get embedding vectors
            embeddings = []
            for i in range(len(emb)):
                embeddings.append(emb[i])
            embeddings = torch.tensor(np.array(embeddings))
            embeddings = embeddings.to(device)
            torch.save(embeddings, 'dataset/'+data_save+'/dw_emb_collab')

    elif args.PE_method == 'LE':  

        if os.path.exists('dataset/'+data_save+'/le_emb_collab'):
            embeddings = torch.load('dataset/'+data_save+'/le_emb_collab',  map_location=torch.device('cpu'))
        else:
            G = nx.from_edgelist(np.array(dataset[0].edge_index).T)
            G = nx.to_scipy_sparse_matrix(G)
            g = dgl.from_scipy(G)
            embeddings = laplacian_positional_encoding(g, args.PE_dim)
            #embeddings = normalize(np.array(embeddings), norm='l2', axis=1, copy=True, return_norm=False)
            embeddings = torch.tensor(embeddings)
            embeddings = embeddings.type(torch.FloatTensor)
            embeddings = embeddings.to(device)
            torch.save(embeddings, 'dataset/'+data_save+'/le_emb_collab')
    embeddings = embeddings.to(device)
    data = data.to(device)
    
    
    with open(f'{args.input_dir}/{args.data_name}/heart_valid_{args.filename}', "rb") as f:
        neg_valid_edge = np.load(f)
        neg_valid_edge = torch.from_numpy(neg_valid_edge)
    with open(f'{args.input_dir}/{args.data_name}/heart_test_{args.filename}', "rb") as f:
        neg_test_edge = np.load(f)
        neg_test_edge = torch.from_numpy(neg_test_edge)
        
    
    split_edge['valid']['edge_neg'] = neg_valid_edge
    split_edge['test']['edge_neg'] = neg_test_edge

    print('train val val_neg test test_neg: ', split_edge['train']['edge'].size(), split_edge['valid']['edge'].size(), split_edge['valid']['edge_neg'].size(), split_edge['test']['edge'].size(), split_edge['test']['edge_neg'].size())


    
    model = PEG(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)
   
    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
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

    print('data adj:', data.adj_t)
    for run in range(args.runs):
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)


        init_seed(seed)

        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)
        best_valid = 0
        kill_cnt = 0
        best_test = 0

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, embeddings, split_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results, score_emb = test(model, predictor, data, embeddings, split_edge, evaluator_hit,evaluator_mrr,
                               args.test_batch_size, args.use_valedges_as_input)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        log_print.info(f'Run: {run + 1:02d}, '
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
                    # if args.save: save_emb(score_emb, save_path)

                
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
        best_auc_valid_str = best_metric_valid_str
        print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))


if __name__ == "__main__":
    main()
