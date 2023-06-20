import os
import time

import torch
from torch.utils.data import DataLoader

from torch_geometric.loader import DataLoader as pygDataLoader
# import wandb
from baseline_models.BUDDY.hashdataset import get_hashed_train_val_test_datasets, make_train_eval_data

def get_loaders(args, dataset, splits, directed):
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
    # if args.model in {'ELPH', 'BUDDY'}:
    train_dataset, val_dataset, test_dataset = get_hashed_train_val_test_datasets(dataset, train_data, val_data,
                                                                                      test_data, args, directed)
   

    dl = DataLoader if args.model in {'ELPH', 'BUDDY'} else pygDataLoader
    train_loader = dl(train_dataset, batch_size=args.batch_size,
                      shuffle=True, num_workers=args.num_workers)
    # as the val and test edges are often sampled they also need to be shuffled
    # the citation2 dataset has specific negatives for each positive and so can't be shuffled
    shuffle_val = False if args.data_name.startswith('ogbl-citation') else True
    val_loader = dl(val_dataset, batch_size=args.batch_size, shuffle=shuffle_val,
                    num_workers=args.num_workers)
    shuffle_test = False if args.data_name.startswith('ogbl-citation') else True
    test_loader = dl(test_dataset, batch_size=args.batch_size, shuffle=shuffle_test,
                     num_workers=args.num_workers)
    if (args.data_name == 'ogbl-citation2') and (args.model in {'ELPH', 'BUDDY'}):
        train_eval_loader = dl(
            make_train_eval_data(args, train_dataset, train_data.num_nodes,
                                  n_pos_samples=5000), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)
    else:
        # todo change this so that eval doesn't have to use the full training set
        train_eval_loader = train_loader

    return train_loader, train_eval_loader, val_loader, test_loader




def get_loaders_hard_neg(args, dataset, splits, directed):
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
    # if args.model in {'ELPH', 'BUDDY'}:
    train_dataset, val_dataset, test_dataset = get_hashed_train_val_test_datasets(dataset, train_data, val_data,
                                                                                      test_data, args, directed)
   

    dl = DataLoader if args.model in {'ELPH', 'BUDDY'} else pygDataLoader
    train_loader = dl(train_dataset, batch_size=args.batch_size,
                      shuffle=True, num_workers=args.num_workers)
    # as the val and test edges are often sampled they also need to be shuffled
    # the citation2 dataset has specific negatives for each positive and so can't be shuffled
    
    val_loader = dl(val_dataset, batch_size=args.test_batch_size, shuffle=False,
                    num_workers=args.num_workers)
    
    test_loader = dl(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                     num_workers=args.num_workers)
    
    train_eval_loader = train_loader

    return train_loader, train_eval_loader, val_loader, test_loader
