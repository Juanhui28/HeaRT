import sys
sys.path.append("..") 
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_auc, evaluate_mrr

# from logger import Logger
from utils import *

import numpy as np
import os

dir_path = '..'
log_print		= get_logger('testrun', 'log', dir_path+'/config/')
def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

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

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(predictor, x, split_edge, optimizer, batch_size):
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        pos_out = predictor(x[edge[0]], x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                             device=x.device)
        neg_out = predictor(x[edge[0]], x[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, x, split_edge, evaluator_hit, evaluator_mrr, batch_size):
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    # pos_train_preds = []
    # for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
    #     edge = pos_train_edge[perm].t()
    #     pos_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    # pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    neg_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]

        neg_edges = torch.permute(neg_valid_edge[perm], (2, 0, 1))
        neg_valid_preds += [predictor(x[neg_edges[0]], x[neg_edges[1]]).squeeze().cpu()]

    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)


    pos_test_preds = []
    neg_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]

        neg_edges = torch.permute(neg_test_edge[perm], (2, 0, 1))
        neg_test_preds += [predictor(x[neg_edges[0]], x[neg_edges[1]]).squeeze().cpu()]


    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = pos_valid_pred

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)
    
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)


    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (MF)')
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--kill_cnt', type=int, default=20)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default='dataset')
    parser.add_argument('--filename', type=str, default='samples_agg-min_norm-1.npy')
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')

    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.data_name)
    split_edge = dataset.get_edge_split()
    data = dataset[0]

    if args.data_name == 'ogbl-collab':
        read_data_name = args.data_name.replace('-', '_')
        with open(f'{args.input_dir}/{read_data_name}/valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{read_data_name}/test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
            
        
        split_edge['valid']['edge_neg'] = neg_valid_edge
        split_edge['test']['edge_neg'] = neg_test_edge
    
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
        split_edge['valid']['edge_neg'] = neg_valid_edge
        split_edge['test']['edge_neg'] = neg_test_edge
        
        split_edge['valid']['edge'] = pos_valid_edge
        split_edge['test']['edge'] = pos_test_edge
        
    


    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                args.num_layers, args.dropout).to(device)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)
   
   
    logger = {
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
    
    
    
    best_valid_auc = best_test_auc = 2
    best_auc_valid_str = 2
    for run in range(args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_numlayer' + str(args.num_layers)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)
        

        emb.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(emb.parameters()) + list(predictor.parameters()), lr=args.lr)
        kill_cnt=0
        best_valid = 0
        
        for epoch in range(1, 1 + args.epochs):
            loss = train(predictor, emb.weight, split_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results, score_emb = test(predictor, emb.weight, split_edge, evaluator_hit, evaluator_mrr,
                               args.test_batch_size)
                for key, result in results.items():
                    logger[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    

                r = torch.tensor(logger[eval_metric].results[run])
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
                        
                        torch.save(optimizer.state_dict(),save_path+'_op')
                        torch.save(emb,save_path+'_emb')
                        torch.save(predictor.state_dict(), save_path+'_predictor')
                        save_emb(score_emb, save_path)

                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break

        for key in logger.keys():
            if len(logger[key].results[0]) > 0:
                print(key)
                logger[key].print_statistics(run)

    # for key in logger.keys():
    #     print(key)
    #     logger[key].print_statistics()
    result_all_run = {}
    for key in logger.keys():
        if len(logger[key].results[0]) > 0:
            print(key)
            
            best_metric,  best_valid_mean, mean_list, var_list = logger[key].print_statistics()

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