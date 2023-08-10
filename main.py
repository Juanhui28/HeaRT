
from utils.evaluation import test

def main():
    args = generate_args()
    args = initialise_wandb(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"executing on {device}")
    algorithm = eval(args.algo_name)(args).to(device)
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    parameters = list(algorithm.parameters())
    total_params = sum(p.numel() for param in parameters for p in param)
    # TODO: add dataset name
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(dataloader_name, args, directed=False)
    results_list = []
    train_func = get_train_func(args)
    for rep in range(args.reps):
        # TODO: add the evaluator selection function
        if args.dataset_name.startswith('ogbl'):  # then this is one of the ogb link prediction datasets
            evaluator = Evaluator(name=args.dataset_name)
        else:
            evaluator = Evaluator(name='ogbl-ppa')  # this sets HR@100 as the metric
        # instance a new algorithm 
        algorithm = eval(args.algo_name)(args).to(device)
        # initialize the algorithm and the corresponding model 
        val_res = test_res = best_epoch = 0
        print(f'running repetition {rep}')
        if rep == 0:
            print_model_params(model)
        for epoch in range(args.epochs):
            t0 = time.time()
            # import ipdb; ipdb.set_trace()
            loss = algorithm.train(args, train_data, optimizer, device)
            if (epoch + 1) % args.eval_steps == 0:
                # TODO: support more evaluator together
                results = test(algorithm, evaluator, train_eval_loader, val_loader, test_loader, args, device,
                               eval_metric=eval_metric)
                for key, result in results.items():
                    train_res, tmp_val_res, tmp_test_res = result
                    if tmp_val_res > val_res:
                        val_res = tmp_val_res
                        test_res = tmp_test_res
                        best_epoch = epoch
                    res_dic = {f'rep{rep}_loss': loss, f'rep{rep}_Train' + key: 100 * train_res,
                               f'rep{rep}_Val' + key: 100 * val_res, f'rep{rep}_tmp_val' + key: 100 * tmp_val_res,
                               f'rep{rep}_tmp_test' + key: 100 * tmp_test_res,
                               f'rep{rep}_Test' + key: 100 * test_res, f'rep{rep}_best_epoch': best_epoch,
                               f'rep{rep}_epoch_time': time.time() - t0, 'epoch_step': epoch}
                    if args.wandb:
                        wandb.log(res_dic)
                    to_print = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_res:.2f}%, Valid: ' \
                               f'{100 * val_res:.2f}%, Test: {100 * test_res:.2f}%, epoch time: {time.time() - t0:.1f}'
                    print(key)
                    print(to_print)
        
        if args.reps > 1:
            results_list.append([test_res, val_res, train_res])
        # report the average performance
        if args.reps > 1:
            test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results_list, axis=0) * 100
            test_acc_std = np.sqrt(np.var(results_list, axis=0)[0]) * 100
            wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                             'test_acc_std': test_acc_std}
            print(wandb_results)
            if args.wandb:
                wandb.log(wandb_results)
        if args.wandb:
            wandb.finish()
        if args.save_model:
            path = f'{ROOT_DIR}/saved_models/{args.dataset_name}'
            torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()
