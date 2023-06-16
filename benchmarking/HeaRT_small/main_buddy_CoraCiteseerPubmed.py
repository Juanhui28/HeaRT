import argparse
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
from baseline_models.BUDDY.data import get_loaders_hard_neg
from baseline_models.BUDDY.utils import select_embedding, select_model, get_num_samples, get_loss, get_split_samples, str2bool
from torch.utils.data import DataLoader
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc


log_print		= get_logger('testrun', 'log', '../config/')

def get_edge_label(pos_edges, neg_edges):

    n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
    edge_label = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=0).t()

    return edge_label, edge_label_index

def get_data(args):

    data_name = args.data_name
    dataset = Planetoid('./dataset', data_name)

    data = read_data(data_name, args.input_dir, args.filename)

    transform = RandomLinkSplit(is_undirected=True, num_val=0.05, num_test=0.1,
                                    add_negative_train_samples=True)
    
    train_data, val_data, test_data = transform(dataset.data)

    train_pos = data['train']['edge'].t()
    train_pos_re = to_undirected(train_pos)

    train_data.edge_index = train_pos_re
    val_data.edge_index = train_pos_re
    test_data.edge_index = train_pos_re
    train_data.edge_label_index[:, train_data.edge_label==1] = train_pos

    valid_pos = data['valid']['edge']
    valid_neg = data['valid']['edge_neg']
    val_data.edge_label,  val_data.edge_label_index = get_edge_label(valid_pos, valid_neg )
   

    test_pos = data['test']['edge']
    test_neg = data['test']['edge_neg']
    test_data.edge_label, test_data.edge_label_index =  get_edge_label(test_pos, test_neg )
  
    directed = False

    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    return dataset, splits, directed

def read_data( data_name, dir_path, filename):
   
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
   
    ##############
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    node_set = set()
    
    
    for split in ['train', 'test', 'valid']:

        path = dir_path + '/{}/{}_pos.txt'.format(data_name, split)

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
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    valid_neg = valid_neg.view(-1, 2)
    test_neg = test_neg.view(-1, 2)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    split_edge['train']['edge'] = train_pos_tensor
    # data['train_val'] = train_val

    split_edge['valid']['edge']= valid_pos
    split_edge['valid']['edge_neg'] = valid_neg
    split_edge['test']['edge']  = test_pos
    split_edge['test']['edge_neg']  = test_neg

    print('train val val_neg test test_neg: ', split_edge['train']['edge'].size(), split_edge['valid']['edge'].size(), split_edge['valid']['edge_neg'].size(), split_edge['test']['edge'].size(), split_edge['test']['edge_neg'].size())


    return split_edge

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
   
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


def train_elph(model, optimizer, train_loader, args, device):
    """
    train a GNN that calculates hashes using message passing
    @param model:
    @param optimizer:
    @param train_loader:
    @param args:
    @param device:
    @return:
    """
   
    t0 = time.time()
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
    for batch_count, indices in enumerate((loader)):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        # get node features
        node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
        curr_links = links[indices].to(device)
        batch_node_features = None if node_features is None else node_features[curr_links]
        batch_emb = None if emb is None else emb[curr_links].to(device)
        # hydrate link features
        if args.use_struct_feature:
            subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
        else:  # todo fix this
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        start_time = time.time()
        optimizer.zero_grad()
        logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        batch_processing_times.append(time.time() - start_time)
   

    return total_loss / len(train_loader.dataset)



def train(model, optimizer, train_loader, args, device, emb=None):
    # print('starting training')
    t0 = time.time()
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
        start_time = time.time()
        optimizer.zero_grad()
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        # batch_processing_times.append(time.time() - start_time)

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_edge(model, loader, device, args, split=None):

    model.eval()
    n_samples = get_split_samples(split, args, len(loader.dataset))
    t0 = time.time()
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
def test_edge_elph(model, loader, device, args, split=None):
    n_samples = get_split_samples(split, args, len(loader.dataset))
    t0 = time.time()
    preds = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    # get node features
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.to(device))
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
    for batch_count, indices in enumerate((loader)):
        curr_links = links[indices].to(device)
        batch_emb = None if emb is None else emb[curr_links].to(device)
        if args.use_struct_feature:
            subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        batch_node_features = None if node_features is None else node_features[curr_links]
        logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break

    
    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    return pos_pred, neg_pred


def get_test_func(model_str):
    if model_str == 'ELPH':
        return test_edge_elph
    elif model_str == 'BUDDY':
        return test_edge
    
@torch.no_grad()
def test(model, evaluator_hit, evaluator_mrr, train_loader, val_loader, test_loader, args, device):

    
    test_func = get_test_func(args.model)

   
    pos_valid_pred, neg_valid_pred  = test_func(model, val_loader, device, args, split='val')
    
    pos_test_pred, neg_test_pred = test_func(model, test_loader, device, args, split='test')

    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred =  torch.flatten(pos_test_pred)

    neg_valid_pred = neg_valid_pred.view(pos_valid_pred.size(0), -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)

    pos_train_pred = pos_valid_pred

    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb



def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    
    ##gnn setting
    
    parser.add_argument('--hidden_channels', type=int, default=256)
    

    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default='dataset')
    parser.add_argument('--filename', type=str, default='samples.npy')
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
    
    parser.add_argument('--no_sf_elph', action='store_true',
                        help='use the structural feature in elph or not')
    parser.add_argument('--feature_prop', type=str, default='gcn',
                        help='how to propagate ELPH node features. Values are gcn, residual (resGCN) or cat (jumping knowledge networks)')
    
    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=4096)

    args = parser.parse_args()
   
    print(args)
    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    dataset, splits, directed = get_data(args)
    train_loader, train_eval_loader, val_loader, test_loader = get_loaders_hard_neg(args, dataset, splits, directed)

    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
       
    }

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
       
        save_path = args.output_dir+'/lr'+str(args.lr) + '_l2'+ str(args.l2) + '_dp' + str(args.feature_dropout) +'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)


        emb = select_embedding(args, dataset.data.num_nodes, device)
        model, optimizer = select_model(args, dataset, emb, device)
        best_valid = 0
        kill_cnt = 0

        for epoch in range(1, 1 + args.epochs):
            if args.model == 'BUDDY':
                loss = train(model, optimizer, train_loader, args, device)
            elif args.model == 'ELPH':
                loss = train_elph(model, optimizer, train_loader, args, device)

            if epoch % args.eval_steps == 0:

                results_rank, score_emb = test(model, evaluator_hit, evaluator_mrr, train_eval_loader, val_loader, test_loader, args, device)

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
    
    print(best_metric_valid_str +' ' +best_auc_valid_str)

    best_auc_metric = best_valid_mean_metric

    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    
    main()

   