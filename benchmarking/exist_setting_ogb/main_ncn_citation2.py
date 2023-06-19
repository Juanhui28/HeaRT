import sys
sys.path.append("..") 

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from baseline_models.NCN.model import predictor_dict, convdict, GCN, DropEdge
from functools import partial

from ogb.linkproppred import Evaluator,PygLinkPropPredDataset

from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils import Logger
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc


def set_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
def save_emb(score_emb, save_path):

    pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
    state = {
    'pos_valid_score': pos_valid_pred,
    'neg_valid_score': neg_valid_pred,
    'pos_test_score': pos_test_pred,
    'neg_test_score': neg_test_pred,
    'node_emb': x
    }

    torch.save(state, save_path)

class PermIterator:

    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret
    
def loaddataset(name, use_valedges_as_input, load=None):
   
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

def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True):
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = []
    adjmask = torch.ones_like(source_edge, dtype=torch.bool)
    for perm in PermIterator(
            source_edge.device, source_edge.shape[0], batch_size
    ): 
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = torch.stack((source_edge[adjmask], target_edge[adjmask]), dim=0)
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   source_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        
        src, dst = source_edge[perm], target_edge[perm]
        pos_out = predictor(h, adj, torch.stack((src, dst)))

        pos_loss = -F.logsigmoid(pos_out).mean()

        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        neg_out = predictor(h, adj, torch.stack((src, dst_neg)))
        neg_loss = -F.logsigmoid(-neg_out).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

def get_metric_score( evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
   
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator_mrr, batch_size):
    model.eval()
    predictor.eval()
    adj = data.full_adj_t
    h = model(data.x, adj)

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in PermIterator(source.device, source.shape[0], batch_size, False):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h, adj, torch.stack((src, dst))).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in PermIterator(source.device, source.shape[0], batch_size, False):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h, adj, torch.stack((src, dst_neg))).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        # mrr = evaluator.eval({
        #     'y_pred_pos': pos_pred,
        #     'y_pred_neg': neg_pred,
        # })['mrr_list'].mean().item()

        
        return pos_pred, neg_pred
    


    train_mrr = 0.0 #test_split('eval_train')
    pos_valid_pred, neg_valid_pred = test_split('valid')
    pos_test_pred, neg_test_pred  = test_split('test')


    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = pos_valid_pred

    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score( evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)


    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(),  h.cpu()]


    return result, h.cpu(), score_emb


def parseargs():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
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
    parser.add_argument('--gnnlr', type=float, default=0.0003)
    parser.add_argument('--prelr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--runs', type=int, default=3)
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
    parser.add_argument("--use_valedges_as_input", action="store_true")
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument('--depth', type=int, default=1)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=30,    type=int,       help='early stopping')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--save', action="store_true")

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
    evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    data, split_edge = loaddataset(args.dataset, False, args.load)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor in ["incn1cn1", "sincn1cn1"]:
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []

    if args.dataset =='collab':
        eval_metric = 'Hits@50'
    elif args.dataset =='ddi':
        eval_metric = 'Hits@20'

    elif args.dataset =='ppa':
        eval_metric = 'Hits@100'
    
    elif args.dataset =='citation2':
        eval_metric = 'MRR'

    kill_cnt = 0
    best_valid = 0
    
    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs)
      
    }
    best_valid_auc = best_test_auc = 2
    best_auc_valid_str = 2

    for run in range(args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        set_seed(seed)
        save_path = args.output_dir+'/lr'+str(args.gnnlr) + '_drop' + str(args.gnndp) +  '_numlayer' + str(args.mplayers)+ '_dim'+str(args.hiddim) + '_'+ 'best_run_'+str(seed)
        
        
        bestscore = [0, 0, 0]
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp).to(device)

        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}])

        kill_cnt = 0
        best_valid = 0

        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput)
            print(f"trn time {time.time()-t1:.2f} s")
            if True:
                t1 = time.time()
                results, h, score_emb = test(model, predictor, data, split_edge, evaluator,
                               args.testbs)
                
                print(f"test time {time.time()-t1:.2f} s")

                
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    train_mrr, valid_mrr, test_mrr = result


                    print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_mrr:.2f}%, '
                            f'Valid: {100 * valid_mrr:.2f}%, '
                            f'Test: {100 * test_mrr:.2f}%')
                    
                    r = torch.tensor(loggers[eval_metric].results[run])
                    best_valid_current = round(r[:, 1].max().item(),4)
                    best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                    print(eval_metric)
                    print(f'best valid: {100*best_valid_current:.2f}%, '
                                f'best test: {100*best_test:.2f}%')

                    print('---', flush=True)

                if best_valid_current >   best_valid:
                    kill_cnt = 0
                    best_valid =  best_valid_current
                    if args.save:
                        torch.save(model.state_dict(), save_path+'_model')
                        torch.save(optimizer.state_dict(),save_path+'_op')
                        torch.save(predictor.state_dict(), save_path+'_predictor')
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
        
   

    if args.runs == 1:
        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))
    
    else:
        print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))
   

if __name__ == "__main__":
    main()