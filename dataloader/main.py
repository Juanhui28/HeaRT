from Data import get_data
from dataloader.functional import ROOT_DIR, get_src_dst_degree, get_pos_neg_edges, get_same_source_negs


def get_train_val_test_datasets(dataloader_name, args, directed=False):
    # root = f'{dataset.root}/elph_'
    # print(f'data path: {root}')
    use_coalesce = True if args.dataset_name == 'ogbl-collab' else False
    dataset, splits, directed, eval_metric = get_data(args)
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    print(
        f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')
    print('constructing training dataset object')
    # import ipdb; ipdb.set_trace()
    train_dataset = eval(args.dataloader_name)(root, 'train', train_data, pos_train_edge, neg_train_edge, args,
                                use_coalesce=use_coalesce, directed=directed).to(device)
    print('constructing validation dataset object')
    val_dataset = eval(args.dataloader_name)(root, 'valid', val_data, pos_val_edge, neg_val_edge, args,
                              use_coalesce=use_coalesce, directed=directed).to(device)
    print('constructing test dataset object')
    test_dataset = eval(args.dataloader_name)(root, 'test', test_data, pos_test_edge, neg_test_edge, args,
                               use_coalesce=use_coalesce, directed=directed).to(device)
    return train_dataset, val_dataset, test_dataset
