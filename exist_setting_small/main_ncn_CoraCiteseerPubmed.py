import argparse
import numpy as np
import torch
import sys
sys.path.append("..") 
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from baseline_models.NCN.model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from baseline_models.NCN.util import PermIterator
import time
# from ogbdataset import loaddataset
from typing import Iterable
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from utils import init_seed, Logger, save_emb, get_logger
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

log_print		= get_logger('testrun', 'log', '../config/')
def randomsplit(dataset, data_name):
   
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
   
    ##############
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    node_set = set()
    dir_path = '.'
    
    for split in ['train', 'test', 'valid']:

        path = dir_path+'/dataset' + '/{}/{}_pos.txt'.format(data_name, split)

       
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

        path = dir_path+'/dataset' + '/{}/{}_neg.txt'.format(data_name, split)

      
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            
            if split == 'valid': 
                valid_neg.append((sub, obj))
               
            if split == 'test': 
                test_neg.append((sub, obj))

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    split_edge['train']['edge'] = train_pos_tensor
    # data['train_val'] = train_val

    split_edge['valid']['edge']= valid_pos
    split_edge['valid']['edge_neg'] = valid_neg
    split_edge['test']['edge']  = test_pos
    split_edge['test']['edge_neg']  = test_neg

    return split_edge

def loaddataset(name, use_valedges_as_input, load=None):
    
    dataset = Planetoid(root="dataset", name=name)
    name = name.lower()
    split_edge = randomsplit(dataset, name)
    data = dataset[0]
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
 
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())
    # if data.edge_weight is None else data.edge_weight.view(-1).to(torch.float)
    # data = T.ToSparseTensor()(data)
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    
    dir_path = '.'
    feature_embeddings = torch.load(dir_path+'/dataset' + '/{}/{}'.format(name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data.x = feature_embeddings
    data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])


    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge



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
def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):
    def penalty(posout, negout):
        scale = torch.ones_like(posout[[0]]).requires_grad_()
        loss = -F.logsigmoid(posout*scale).mean()-F.logsigmoid(-negout*scale).mean()
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(torch.square(grad))
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr, batch_size,
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    # pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    '''
    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    '''

    pos_valid_pred = torch.cat([
        predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred = torch.cat([
        predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    
    print('train valid_pos valid_neg test_pos test_neg', pos_valid_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]

    return result, score_emb


def parseargs():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--mplayers', type=int, default=1)
    parser.add_argument('--nnlayers', type=int, default=3)
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--lnnn', action="store_true")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--jk', action="store_true")
    parser.add_argument('--maskinput', action="store_true")
    parser.add_argument('--hiddim', type=int, default=32)
    parser.add_argument('--gnndp', type=float, default=0.3)
    parser.add_argument('--xdp', type=float, default=0.3)
    parser.add_argument('--tdp', type=float, default=0.3)
    parser.add_argument('--gnnedp', type=float, default=0.3)
    parser.add_argument('--predp', type=float, default=0.3)
    parser.add_argument('--preedp', type=float, default=0.3)
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument('--gnnlr', type=float, default=0.0003)
    parser.add_argument('--prelr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="collab")
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument('--model', choices=convdict.keys())
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument('--save_gemb', action="store_true")
    parser.add_argument('--load', type=str)
    parser.add_argument('--cnprob', type=float, default=0)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument("--savex", action="store_true")
    parser.add_argument("--loadx", action="store_true")
    parser.add_argument("--loadmod", action="store_true")
    parser.add_argument("--savemod", action="store_true")


    ###
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--eval_steps', type=int, default=5)
    
    args = parser.parse_args()
    return args


def main():
    args = parseargs()



    print(args, flush=True)
  
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
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


    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]

    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []
    
    for run in range(0, args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)

        save_path = args.output_dir+'/lr'+str(args.gnnlr) + '_drop' + str(args.gnndp) + '_l2'+ str(args.l2) + '_numlayer' + str(args.mplayers)+ '_numPredlay' + str(args.nnlayers) +'_dim'+str(args.hiddim) + '_'+ 'best_run_'+str(seed)


        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
       
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
       
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}],  weight_decay=args.l2)
        
        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            # print(f"trn time {time.time()-t1:.2f} s", flush=True)
            
            t1 = time.time()
            if epoch % args.eval_steps == 0:
                results, score_emb = test(model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr,
                                args.testbs, args.use_valedges_as_input)
                # print(f"test time {time.time()-t1:.2f} s")
            
                
                for key, result in results.items():
                    _, valid_hits, test_hits = result

                   
                    loggers[key].add_result(run, result)
                        
                    print(key)
                    log_print.info(
                        f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---', flush=True)

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max().item()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save:

                        save_emb(score_emb, save_path)
        
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
  