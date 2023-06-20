import sys
sys.path.append("..") 

import os, torch, dgl
import argparse

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from baseline_models.PEG.Graph_embedding import DeepWalk
from baseline_models.PEG.utils import laplacian_positional_encoding
from baseline_models.PEG.model import Net
from torch.utils.data import DataLoader

import networkx as nx
import scipy.sparse as sp

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from utils import *
from torch_sparse import SparseTensor
import scipy.sparse as ssp

dir_path = get_root_dir()
log_print		= get_logger('testrun', 'log', get_config_dir())


def read_data(data_name):
    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

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
    data['edge_index'] = edge_index

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

def get_10fold(features, train_pos, train_matrix, args, device):
    train_pos = train_pos.numpy().tolist()
    random.shuffle(train_pos)

    slice_num = int(len(train_pos)/10)
    train_pos_slice = [train_pos[i:i+slice_num] for i in range(0,len(train_pos ),slice_num)]

    pipe_train_x_list = []
    pipe_train_edge_index_list = []

    for j in range(10):
        print(j)
        id_train_pos = train_pos_slice[j]

        pipe_train_matrix = np.copy(train_matrix)
        pipe_train_matrix[np.array(id_train_pos).T[0],np.array(id_train_pos).T[1]] = 0
        pipe_train_matrix[np.array(id_train_pos).T[1],np.array(id_train_pos).T[0]] = 0

        if args.PE_method == 'DW':
            #deepwalk
            G = nx.DiGraph(pipe_train_matrix)
            model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
            model_emb.train(embed_size = args.PE_dim)# train model
            emb = model_emb.get_embeddings()# get embedding vectors
            embeddings = []
            for i in range(len(emb)):
                embeddings.append(emb[i])
            embeddings = np.array(embeddings)
        elif args.PE_method == 'LE':
        
            #LAP
            sp_adj = sp.coo_matrix(pipe_train_matrix)
            g = dgl.from_scipy(sp_adj)
            embeddings = np.array(laplacian_positional_encoding(g, args.PE_dim))
            embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)

        pipe_train_edge_index = [i for i in train_pos if i not in id_train_pos]
        pipe_train_x = torch.cat((torch.tensor(embeddings), features), 1)

        pipe_edge_index = np.array(pipe_train_edge_index).transpose()
        pipe_edge_index = torch.from_numpy(pipe_edge_index)
        
        pipe_train_x = pipe_train_x.unsqueeze_(0)
        pipe_edge_index = pipe_edge_index.unsqueeze_(0)


        pipe_train_x_list.append(pipe_train_x)
        pipe_train_edge_index_list.append(pipe_edge_index)

        
    pipe_train_x = torch.cat(pipe_train_x_list, dim=0).to(device)
    pipe_edge_index = torch.cat(pipe_train_edge_index_list, dim=0).to(device)

    pipe_train_x = pipe_train_x.cuda(device)
    pipe_edge_index = pipe_edge_index.cuda(device)

    return pipe_train_x, pipe_edge_index, train_pos_slice

def train(model, optimizer, x, train_pos, train_pos_loss, batch_size):
    model.train()
    m = torch.nn.Sigmoid()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    num_nodes = x.size(0)
    train_pos = train_pos.to(x.device)

    for perm in DataLoader(range(train_pos_loss.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge_index = train_pos.t()

        edge = train_pos[perm].t()

        h = model.get_emb(x, edge_index)
        output = model.score(h, edge)
        pos_out = m(output)
        pos_out = torch.squeeze(pos_out)
        pos_loss = -torch.log(pos_out + 1e-15).mean()


        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        
        output = model.score(h, edge)
        neg_out = m(output)
        neg_out = torch.squeeze(neg_out)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        with torch.no_grad():
            model.fc.weight[0][0].clamp_(1e-5,100)

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test_edge(model, input_data, h, batch_size):

   
    preds = []
    input_data = input_data.to(h.device)
    m = torch.nn.Sigmoid()
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        score = model.score(h, edge).cpu()

        preds += [m(score)]
        
    pred_all = torch.cat(preds, dim=0)

    return pred_all

torch.no_grad()
def test(model, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()

    # adj_t = adj_t.transpose(1,0)
    
    edge_index = data['edge_index']
    edge_index = edge_index.to(x.device)
    
    h = model.get_emb(x, edge_index)
    
    x = h

    pos_train_pred = test_edge(model, data['train_val'], h, batch_size)

    neg_valid_pred = test_edge(model, data['valid_neg'], h, batch_size)

    pos_valid_pred = test_edge(model, data['valid_pos'], h, batch_size)

    pos_test_pred = test_edge(model, data['test_pos'], h, batch_size)

    neg_test_pred = test_edge(model, data['test_neg'], h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]

    return result, score_emb


def main():
    parser = argparse.ArgumentParser(description='homo')

    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--data_name', type=str, default='cora')

    parser.add_argument('--PE_dim', type=int, default=128, help = 'dimension of positional encoding')
    parser.add_argument('--hidden_dim', type=int, default=128, help = 'hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)

    parser.add_argument('--random_partition', action='store_true', help = 'whether to use random partition while training')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--feature_type', type=str, default="N", help = 'features type, N means node feature, C means constant feature (node degree)',
                    choices = ['N', 'C'])
    parser.add_argument('--PE_method', type=str, default="LE", help = 'positional encoding techniques',
                    choices = ['DW', 'LE'])
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--save', action='store_true', default=False)

    ### debug
    # parser.add_argument('--device', type=int, default=2)
    # parser.add_argument('--random_partition', action='store_true', default=False,help = 'whether to use random partition while training')
    parser.add_argument('--no_pe', action='store_true', help = 'whether to use pe')

    args = parser.parse_args()
    
    print(args)

    # [115, 105, 100]
    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    device = args.device

    data = read_data(args.data_name)
    adj = data['adj'].to_dense()
    train_matrix=np.copy(adj)
    features = data['x']


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

   

    

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        
        init_seed(seed)

        
        if args.feature_type == 'N':
            pca=PCA(n_components=args.hidden_dim)
            features=pca.fit_transform(np.array(features))
            features = torch.tensor(features)
            features = features.type(torch.FloatTensor)

        if args.PE_method == 'DW':
            #deepwalk
            G = nx.DiGraph(train_matrix)
            model_emb = DeepWalk(G,walk_length=80, num_walks=10,workers=1)#init model
            model_emb.train(embed_size = args.PE_dim)# train model
            emb = model_emb.get_embeddings()# get embedding vectors
            embeddings = []
            for i in range(len(emb)):
                embeddings.append(emb[i])
            embeddings = np.array(embeddings)

        elif args.PE_method == 'LE':
            #LAP
            sp_adj = sp.coo_matrix(train_matrix)
            g = dgl.from_scipy(sp_adj)
            embeddings = np.array(laplacian_positional_encoding(g, args.PE_dim))
            embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)

        x = torch.cat((torch.tensor(embeddings), features), 1)
        # edge_index = np.array(train_edge_index).transpose()
        # edge_index = torch.from_numpy(edge_index)
        
        x = x.to(device)
        # edge_index = edge_index.cuda(device)

        model = Net(in_feats_dim = len(features[1]), pos_dim = args.PE_dim, hidden_dim = args.hidden_dim, no_pe=args.no_pe)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters() ,lr=args.lr, weight_decay=args.l2)
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_l2'+ str(args.l2)  +'_dim'+str(args.hidden_dim) + '_'+ 'best_run_'+str(seed)

        best_valid = 0
        kill_cnt = 0

        if args.random_partition:
            pipe_train_x, pipe_edge_index, positive_train = get_10fold(features, data['train_pos'], train_matrix, args, device)

        small_epoch_list = []
        for i in range(2):
            small_epoch_list.append(i)

        for epoch in range(1, 1 + args.epochs):

            if args.random_partition:
                
                result_epoch = {}
                results_rank = {}
                random.shuffle(small_epoch_list)
                for small_epoch in small_epoch_list:
                    x = pipe_train_x[small_epoch]
                    edge = pipe_edge_index[small_epoch].t()
                    train_pos = torch.tensor(positive_train[small_epoch])
                    loss = train(model, optimizer, x, edge, train_pos, args.batch_size)

                    if epoch % args.eval_steps == 0:
                        results_multi, score_emb = test(model, data, x, evaluator_hit, evaluator_mrr, args.batch_size)

                        for key, result in results_multi.items():
                            if key not in result_epoch: result_epoch[key] = []
                            result_epoch[key].append(result)
                    
                if epoch % args.eval_steps == 0:
                    for key, result_list in result_epoch.items():
                        ave_result_train = torch.tensor(result_list)[:,0].mean().item()
                        ave_result_valid = torch.tensor(result_list)[:,1].mean().item()
                        ave_result_test = torch.tensor(result_list)[:,2].mean().item()
                        
                        results_rank[key] = (ave_result_train, ave_result_valid, ave_result_test)
                        loggers[key].add_result(run, (ave_result_train, ave_result_valid, ave_result_test))
                

            else:
                loss = train(model, optimizer, x, data['train_pos'],  data['train_pos'], args.batch_size)

                if epoch % args.eval_steps == 0:
                    results_rank, score_emb = test(model, data, x, evaluator_hit, evaluator_mrr, args.batch_size)

                    for key, result in results_rank.items():
                        loggers[key].add_result(run, result)

            if epoch % args.eval_steps == 0 :
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
        
    
    print(best_metric_valid_str +' ' +best_auc_valid_str)

    return best_valid_mean_metric, best_auc_metric, result_all_run


if __name__ == "__main__":
    main()
   