import sys
sys.path.append("..") 
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from baseline_models.NCN.model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_geometric.datasets import Planetoid
from torch.utils.tensorboard import SummaryWriter
from baseline_models.NCN.util import PermIterator
from utils import init_seed, Logger, save_emb, get_logger, get_config_dir, get_data_dir
import time
from typing import Iterable
import random
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc


    
def set_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

log_print		= get_logger('testrun', 'log', get_config_dir())
def randomsplit(dataset, data_name, dir_path, filename, val_ratio=0.1, test_ratio=0.2):


    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    node_set = set()
   
    data_name = data_name.lower()
    for split in ['train', 'test', 'valid']:

        path = dir_path+ '/{}/{}_pos.txt'.format(data_name, split)

       
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

    with open(f'{dir_path}/{data_name}/valid_{filename}', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/test_{filename}', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)

    test_pos =  torch.tensor(test_pos)

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

def loaddataset(name, use_valedges_as_input,  dir_path, filename, load=None):

    if name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset, name,  dir_path, filename)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
        feature_embeddings = torch.load(dir_path+ '/{}/{}'.format(name, 'gnn_feature'))
        feature_embeddings = feature_embeddings['entity_embedding']

        data.x = feature_embeddings
    else:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        edge_index = data.edge_index
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())
    # if data.edge_weight is None else data.edge_weight.view(-1).to(torch.float)
    # data = T.ToSparseTensor()(data)
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()

    

    data.max_x = -1
    if name == "ppa":
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    elif name == "ddi":
        #data.x = torch.zeros((data.num_nodes, 1))
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
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

def get_metric_score( evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
   
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


def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    
    valid_auc = round(valid_auc, 4)
    # test_auc = round(test_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    # test_ap = average_precision_score(test_true, test_pred)
    
    valid_ap = round(valid_ap, 4)
    # test_ap = round(test_ap, 4)
    
    results['AP'] = valid_ap


    return results


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator_mrr, batch_size,
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
    neg_num = neg_test_edge.size(1)

    '''
    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    '''
    pos_preds = []
    neg_preds = []

    for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False):
        
        pos_preds += [predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()]

        neg_edges = torch.permute(neg_valid_edge[perm], (2, 0, 1))
        neg_edges = neg_edges.view(2,-1)
        neg_preds += [predictor(h, adj, neg_edges).squeeze().cpu()]

    pos_valid_pred = torch.cat(pos_preds, dim=0)
    neg_valid_pred = torch.cat(neg_preds, dim=0)
    
    pos_preds = []
    neg_preds = []

    for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False):
        pos_preds += [predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()]
        
        neg_edges = torch.permute(neg_test_edge[perm], (2, 0, 1))
        neg_edges = neg_edges.view(2,-1)
        neg_preds += [predictor(h, adj, neg_edges).squeeze().cpu()]
        
    pos_test_pred = torch.cat(pos_preds, dim=0)
    neg_test_pred = torch.cat(neg_preds, dim=0)

    neg_valid_pred = neg_valid_pred.view(-1, neg_num)
    neg_test_pred = neg_test_pred.view(-1, neg_num)

    
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = pos_valid_pred

    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score( evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]


    return result, h.cpu(), score_emb


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
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="cora")
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

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=30,    type=int,       help='early stopping')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, default=get_data_dir())
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')

    # parser.add_argument('--model', default='puregcn')
    # parser.add_argument('--predictor', default='cn1')
    # parser.add_argument('--device', type=int, default=2)
    

  
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
   
    print(args, flush=True)
    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    # writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    # writer.add_text("hyperparams", hpstr)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.input_dir, args.filename, args.load)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]

    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []
    ret_auc = []

    if args.dataset =='collab':
        eval_metric = 'Hits@50'
    elif args.dataset =='ddi':
        eval_metric = 'Hits@20'

    elif args.dataset =='ppa':
        eval_metric = 'Hits@100'
    
    elif args.dataset =='citation2':
        eval_metric = 'MRR'

    else:  eval_metric = 'MRR'
    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)
    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
    }
   

    for run in range(0, args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        set_seed(seed)

        save_path = args.output_dir+'/lr'+str(args.gnnlr) + '_drop' + str(args.gnndp) +'_l2'+str(args.l2) +  '_numlayer' + str(args.mplayers)+ '_Prednum'+str(args.nnlayers) +'_dim'+str(args.hiddim) + '_'+ 'best_run_'+str(seed)
        
        bestscore = None
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        if args.loadx:
            with torch.no_grad():
                model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
            model.xemb[0].weight.requires_grad_(False)
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        if args.loadmod:
            keys = model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
        
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}], weight_decay=args.l2)
        kill_cnt = 0
        best_valid = 0

        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            # print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if epoch % args.eval_steps == 0:
                t1 = time.time()
                results, h, score_emb = test(model, predictor, data, split_edge, evaluator_mrr,
                               args.testbs, args.use_valedges_as_input)
                # print(f"test time {time.time()-t1:.2f} s")
                if bestscore is None:
                    bestscore = {key: list(results[key]) for key in results}
                

                if True:
                    for key, result in results.items():
                        loggers[key].add_result(run, result)

                        train_hits, valid_hits, test_hits = result
                        if key == eval_metric:
                            current_valid_eval = valid_hits

                        if valid_hits > bestscore[key][1]:
                            bestscore[key] = list(result)
                            if args.save_gemb:
                                torch.save(h, f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                            if args.savex:
                                torch.save(model.xemb[0].weight.detach(), f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            if args.savemod:
                                torch.save(model.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                                torch.save(predictor.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
                        
                        print(key)
                        log_print.info(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                        
                    print(f'best {eval_metric}, '
                            f': {bestscore[eval_metric][2]:.4f}%, ')
                        
        
                    print('---', flush=True)

                if bestscore[eval_metric][1] >   best_valid:
                    kill_cnt = 0
                    best_valid =  bestscore[eval_metric][1]
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
        
    best_auc_valid_str = best_metric_valid_str
    best_auc_metric = best_valid_mean_metric
   
    print(best_metric_valid_str+ ' ' + best_auc_valid_str)

    return best_valid_mean_metric, best_auc_metric, result_all_run

 

       

if __name__ == "__main__":
    main()
   