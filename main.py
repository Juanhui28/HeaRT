
from utils.args import generate_args
from utils.functional import *
from ogb.linkproppred import Evaluator
import torch, os
from dataloader.read_data import read_data
# from model.MOE_model import MOE
# from model.MOE_model_fsdp import MOE

from torch.utils.data import DataLoader
from utils.evaluation import evaluate_hits, evaluate_mrr
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
# from model.MOE_model import MOE
from torch.nn import BCEWithLogitsLoss


log_print		= get_print_logger()


def get_metric_score(args, evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    result = {}
    k_list = [1, 3, 10, 20, 50, 100]
    if args.data_name == 'cora' or args.data_name == 'citeseer' or  args.data_name == 'pubmed' or 'citation2' in args.data_name:
        result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1), k_list)
        result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1), k_list )
        result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1), k_list )
        
        # result_mrr = {}
        result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])

    if 'citation2' not in  args.data_name:
        
        
        result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
        result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
        result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

        for K in k_list:
            result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    if 'citation2'  in  args.data_name:
        for K in k_list:
            result[f'mrr_hit{K}'] = (result_mrr_train[f'Hits{K}'], result_mrr_val[f'Hits{K}'], result_mrr_test[f'Hits{K}'])

    
    return result

def train(args, model, lpdata, optimizer, device_list):
    
    model.train()

    train_pos = lpdata.split_edge['train']['edge']
    train_neg = lpdata.split_edge['train']['edge_neg']


    total_loss = total_examples = 0
    total_balan_loss = 0
    total_bceloss = 0
    for perm in DataLoader(range(train_pos.size(0)), args.batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        # pos edges
        edge = train_pos[perm]
        pos_out, balance_loss, _, _, _ = model(node_feature_pos[perm], struct_feature_pos[perm], edge, perm, 'train')
        num_examples = pos_out.size(0)

        pos_loss = -torch.log(pos_out + 1e-15).mean() + balance_loss
        total_balan_loss += balance_loss*num_examples
        total_bceloss += -torch.log(pos_out + 1e-15).sum()
       
        # neg edges
        edge = train_neg[perm]
        neg_out, balance_loss, _, _, _ = model(node_feature_neg[perm], struct_feature_neg[perm], edge, perm+train_pos.size(0), 'train')
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean() + balance_loss
        total_balan_loss += balance_loss*num_examples
        total_bceloss += -torch.log(1 - neg_out + 1e-15).sum()

        loss = pos_loss + neg_loss
        # try:
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       
        optimizer.step()

        
        total_loss += loss.item() * num_examples
        total_examples += num_examples


    return total_loss / total_examples, total_bceloss/total_examples, total_balan_loss/total_examples



@torch.no_grad()
def test_edge(args, model, node_feature, struct_feature, input_edge, mode, neognn_A,pos_num=None, test_all=False ):

    preds = []
    gating_inds=[]
    gating_scores = []
    expert_scores = []

    if 'neg' in mode:
        if 'test' in mode: mode = 'test'
        elif 'valid' in mode: mode= 'valid'
    
    for perm  in DataLoader(range(input_edge.size(0)), args.testbs, shuffle=False):

        edge = input_edge[perm]
        if pos_num != None:
            tmpperm = perm + pos_num
        else:
            tmpperm = perm
        score, _, gating_score, gating_ind, expert_score= model(node_feature[perm], struct_feature[perm], edge, tmpperm, mode, neognn_A, test_all=test_all)
        
        preds += [score.cpu()]
        gating_inds.append(gating_ind)
        gating_scores.append(gating_score)
        expert_scores.append(expert_score)

    pred_all = torch.cat(preds, dim=0)
    gating = [gating_scores, gating_inds, expert_scores ]
    return pred_all, gating


@torch.no_grad()
def test(args, model, lpdata, evaluator_hit, evaluator_mrr, device_list, epoch):
    model.eval()
   
    if args.neognn:
        if args.use_valedges_as_input:
            A2 = lpdata.full_A2
        else:
            A2 = lpdata.A2
        neognn_A = get_neognn_A(args, A2, model.neognn, model.neognn.f_edge[0].weight.device, lpdata.num_nodes)
    else:
        neognn_A = None
    
    edge = lpdata.split_edge['test']['edge'].to(device_list[0])
    test_pos_num = edge.size(0)
    # if epoch == 5:
    # import ipdb
    # ipdb.set_trace()
    node_feature = lpdata.heuristic_feature['test']['edge'][:, :lpdata.num_features].to(device_list[0])
    struct_feature = lpdata.heuristic_feature['test']['edge'][:, lpdata.num_features:].to(device_list[0])
    pos_test_pred, gating_test_pos = test_edge(args, model, node_feature, struct_feature, edge, 'test', neognn_A)

    edge = lpdata.split_edge['test']['edge_neg'].to(device_list[0])
    node_feature = lpdata.heuristic_feature['test']['edge_neg'][:, :lpdata.num_features].to(device_list[0])
    struct_feature = lpdata.heuristic_feature['test']['edge_neg'][:, lpdata.num_features:].to(device_list[0])
    neg_test_pred, gating_test_neg = test_edge(args, model, node_feature, struct_feature, edge, 'test_neg', neognn_A, test_pos_num)


    edge = lpdata.split_edge['valid']['edge'].to(device_list[0])
    valid_pos_num = edge.size(0)
    node_feature = lpdata.heuristic_feature['valid']['edge'][:, :lpdata.num_features].to(device_list[0])
    struct_feature = lpdata.heuristic_feature['valid']['edge'][:, lpdata.num_features:].to(device_list[0])
    pos_valid_pred, gating_valid_pos = test_edge(args, model, node_feature, struct_feature, edge, 'valid', neognn_A)


    edge = lpdata.split_edge['valid']['edge_neg'].to(device_list[0])
    node_feature = lpdata.heuristic_feature['valid']['edge_neg'][:, :lpdata.num_features].to(device_list[0])
    struct_feature = lpdata.heuristic_feature['valid']['edge_neg'][:, lpdata.num_features:].to(device_list[0])
    neg_valid_pred, gating_valid_neg = test_edge(args, model, node_feature, struct_feature, edge, 'valid_neg', neognn_A, valid_pos_num)

    
   
    if args.use_train_val:
        edge = lpdata.split_edge['train_val']['edge'].to(device_list[0])
        node_feature = lpdata.heuristic_feature['train_val']['edge'][:, :lpdata.num_features].to(device_list[0])
        struct_feature = lpdata.heuristic_feature['train_val']['edge'][:, lpdata.num_features:].to(device_list[0])
        pos_train_pred, gating_train_val = test_edge(args, model, node_feature, struct_feature, edge, 'train_val', neognn_A)

    else:
        pos_train_pred = pos_valid_pred
    # pos_train_pred = torch.flatten(pos_train_pred)
    # neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    # pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    # if epoch  == 5:
    #     import ipdb
    #     ipdb.set_trace()


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(args, evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    # import ipdb
    # ipdb.set_trace()
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb

    
def main():
    args = generate_args()
   
    device = set_device(args)
    # print(f"executing on {device}")

    lpdata, prepdata = read_data(args)
    if args.model.lower() == 'seal':
        seal_train_dataset = prepdata[1]['train']
    else:
        seal_train_dataset = None


    results_list = []
    # train_func = get_train_func(args)

    loggers, eval_metric = get_metric_logger(args)

  
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
     
    for run in range(args.runs):
        if args.runs > 1:
            seed = run
        else:
            seed = args.seed
        init_seed(seed)
        print('seed: ', seed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) +'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        


        model = eval(args.model)(lpdata, baseline_datast, args, seal_train_dataset, device)

        if args.load_dict:
            print('load saved model')
            path = 'output/cora/check_train100/lr0.01_drop0.1_l20.0001_numlayer3_numPredlay2_dim128_best_run_0_model'
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state)
    

        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.l2)

        total_params = sum(p.numel() for param in parameters for p in param)
        
        best_valid  = kill_cnt = 0
        best_epoch = 0
        print(f'running repetition {run}')

        results, score_emb, score_sepa = test(args, model, lpdata, evaluator_hit, evaluator_mrr, device_list, 0)
        
        for key, result in results.items():
            if key in loggers:
                print(key)
                
                train_hits, valid_hits, test_hits = result
                log_print.info(
                            f'Run: {run + 1:02d}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---')
        
        for epoch in range(args.epochs):
            
            if not args.load_dict:
                loss, bceloss, balanceloss = train(args, model, lpdata, optimizer, device_list)
            else:
                loss = 0
            if epoch % args.eval_steps == 0:
               
                
                results, score_emb, score_sepa = test(args, model, lpdata, evaluator_hit, evaluator_mrr, device_list, epoch)
                
                for key, result in results.items():
                    if key in loggers:
                        loggers[key].add_result(run, result)


                for key, result in results.items():
                    if key in loggers:
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                        # import ipdb
                        # ipdb.set_trace()
                        log_print.info(
                                f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'bceLoss: {bceloss:.4f}, '
                                f'balLoss: {balanceloss.item():.4f}, '
                                f'best epoch: {best_epoch}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                        print('---')
                
                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    best_epoch = epoch
                    if args.save:
                        torch.save(model.state_dict(), save_path+'_model')
                        torch.save(optimizer.state_dict(),save_path+'_op')
                        save_emb(score_emb, save_path)
                        torch.save(score_sepa, save_path+'_expert_scores')

                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
            
        for key in loggers.keys():
            if key in loggers:
                print(key)
                loggers[key].print_statistics(run)
    
    # result_all_run = {}
    for key in loggers.keys():
        if key in loggers:
            print(key)
            loggers[key].print_statistics()

       
 

            
        
      
if __name__ == '__main__':
    main()
