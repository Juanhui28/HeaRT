
# from dataloader.functional import ROOT_DIR, get_src_dst_degree, get_pos_neg_edges, get_same_source_negs
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from .Datasets.LPDataset import LPDataset
from .Datasets.SealDataset import *
from .Datasets.BuddyDataset import get_hashed_train_val_test_datasets
from torch.utils.data import DataLoader

def get_buddy_dataset(args, lpdata):
    train_data, val_data, test_data = lpdata.splits['train'], lpdata.splits['valid'], lpdata.splits['test']
    
    if args.use_train_val:
        train_val_dataset =  lpdata.splits['train_val']
    else: train_val_dataset = None

    root = lpdata.dir_path+'/dataset/'+args.data_name.replace('-','_') +'/'
    
    train_dataset, val_dataset, test_dataset, train_val_dataset = get_hashed_train_val_test_datasets(root, train_data, val_data,
                                                                                    test_data, train_val_dataset, lpdata.args, lpdata.directed)

    data = {'train':train_dataset, 'valid': val_dataset, 'test': test_dataset, 'train_val': train_val_dataset}
    return data
    
def get_seal_dataset(args, lpdata):

    args.data_appendix = '_h{}_{}_rph{}'.format(args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))

    path = 'dataset/'+ str(args.data_name)+ '_seal{}'.format(args.data_appendix)
    use_coalesce = True if args.data_name == 'ogbl-collab' else False

    dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
    train_dataset = eval(dataset_class)(
        path, 
        lpdata, 
        lpdata.split_edge, 
        num_hops=args.num_hops, 
        percent=args.train_percent, 
        split='train', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=lpdata.directed, 
        use_valedges_as_input=args.use_valedges_as_input
    ) 
    
    if args.use_train_val:

        train_val_dataset = eval(dataset_class)(
        path, 
        lpdata, 
        lpdata.split_edge, 
        num_hops=args.num_hops, 
        percent=args.val_percent, 
        split='train_val', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=lpdata.directed, 
        use_valedges_as_input=args.use_valedges_as_input
    )
    else: train_val_dataset = None

    dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
    val_dataset = eval(dataset_class)(
        path, 
        lpdata, 
        lpdata.split_edge, 
        num_hops=args.num_hops, 
        percent=args.val_percent, 
        split='valid', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=lpdata.directed, 
        use_valedges_as_input=args.use_valedges_as_input
    )
   
    dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
    test_dataset = eval(dataset_class)(
        path, 
        lpdata, 
        lpdata.split_edge, 
        num_hops=args.num_hops, 
        percent=args.test_percent, 
        split='test', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=lpdata.directed, 
        use_valedges_as_input=args.use_valedges_as_input
    )

    data = {'train':train_dataset, 'valid': val_dataset, 'test': test_dataset, 'train_val': train_val_dataset}
    return data

def get_data_loader(args, lpdata, prepdata):


    if args.model.lower() == 'seal':

        train_loader = DataLoader(prepdata['train'], batch_size=args.trainbs, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(prepdata['valid'], batch_size=args.testbs, num_workers=args.num_workers)
        test_loader = DataLoader(prepdata['test'], batch_size=args.testbs,   num_workers=args.num_workers)
        if args.use_train_val:
            train_val_loader = DataLoader(prepdata['train_val'], batch_size=args.testbs, num_workers=args.num_workers)
        else:
            train_val_loader = None
        dataloader = {'train': train_loader, 'valid':val_loader, 'test': test_loader, 'train_val': train_val_loader}

    elif args.model.lower() == 'buddy':
       
        train_loader = DataLoader(range(len(prepdata['train'].links)), args.trainbs, shuffle=True)
        val_loader = DataLoader(range(len(prepdata['valid'].links)), args.testbs, shuffle=False)
        test_loader = DataLoader(range(len(prepdata['test'].links)), args.testbs, shuffle=False)
        if args.use_train_val:
            train_val_loader = DataLoader(range(len(prepdata['train_val'].links)), args.testbs, shuffle=False)
        else:
            train_val_loader = None
        dataloader = {'train': train_loader, 'valid':val_loader, 'test': test_loader, 'train_val': train_val_loader}

    
    else:
       
        train_loader = DataLoader(range(len(lpdata.split_edge['train']['edge'])), args.trainbs, shuffle=True)

        val_pos_loader = DataLoader(range(len( lpdata.split_edge['valid']['edge'])), args.testbs)
        val_neg_loader = DataLoader(range(len( lpdata.split_edge['valid']['edge_neg'])), args.testbs)
        test_pos_loader = DataLoader(range(len( lpdata.split_edge['test']['edge'])), args.testbs)
        test_neg_loader = DataLoader(range(len( lpdata.split_edge['test']['edge_neg'])), args.testbs)
    
        if args.use_train_val:
            train_val_loader = DataLoader(range(len(lpdata.split_edge['train_val']['edge'])), args.testbs, shuffle=False)
        else:
            train_val_loader = None

        dataloader = {'train': train_loader, 'valid_pos':val_pos_loader, 'valid_neg':val_neg_loader, 'test_pos': test_pos_loader, 'test_neg': test_neg_loader, 'train_val': train_val_loader}

    return dataloader


def read_data(args):

    lpdata = LPDataset(args)


    if args.model.lower() == 'seal':
        prepdata = get_seal_dataset(args, lpdata)
       
    elif args.model.lower() == 'buddy':
        prepdata = get_buddy_dataset(args, lpdata)
    
    else:
        prepdata = None

   
    return lpdata, prepdata
   
