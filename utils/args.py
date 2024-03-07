    
    
import argparse
from distutils.util import strtobool

# TODO: maybe need a function to add some model related args.
def generate_args():
    default_device = 1
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--trainbs', type=int, default=1024)
    parser.add_argument('--testbs', type=int, default=8192, help='test batch size')
  
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
 
    parser.add_argument('--device', type=int, default=default_device)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_train_val', action='store_true', default=False, help='sample some training edges to do evaluation')
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False, help='add validation edges in test on collab')


    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ###### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    
    ###neognn
    parser.add_argument('--f_edge_dim', type=int, default=8) 
    parser.add_argument('--f_node_dim', type=int, default=128) 
    parser.add_argument('--g_phi_dim', type=int, default=128) 
    # parser.add_argument('--beta', type=float, default=0.1)

    ####seal
    parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--test_percent', type=float, default=100)
    parser.add_argument('--num_hops', type=int, default=3)
    parser.add_argument('--node_label', type=str, default='drnl',  help="which specific labeling trick to use")
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--use_feature_seal', action='store_true', 
                    help="whether to use raw node features as GNN input")
    parser.add_argument('--train_node_embedding', action='store_true', 
                    help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
    parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
    


    ### ncn
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--mplayers', type=int, default=1)
    parser.add_argument('--gnndp', type=float, default=0.3)
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--convmodel',  default='puregcn')
    parser.add_argument('--jk', action="store_true")
    parser.add_argument('--gnnedp', type=float, default=0.3)
    parser.add_argument('--xdp', type=float, default=0.3)
    parser.add_argument('--tdp', type=float, default=0.3)
    parser.add_argument("--loadx", action="store_true")
    parser.add_argument('--nnlayers', type=int, default=3)
    parser.add_argument('--predp', type=float, default=0.3)
    parser.add_argument('--preedp', type=float, default=0.3)
    parser.add_argument('--lnnn', action="store_true")
    parser.add_argument('--predictor',  default='cn1')
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument("--learnpt", action="store_true")
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--maskinput', action="store_true")
    

    ###buddy
    parser.add_argument('--use_feature_buddy', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    parser.add_argument('--sign_dropout', type=float, default=0.5)
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
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
    parser.add_argument('--use_struct_feature', type=str2bool, default=True,
                        help="whether to use structural graph features as GNN input")
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    
    #### peg
    parser.add_argument('--PE_method', type=str, default='LE')
    parser.add_argument('--PE_dim', type=int, default=128)

  
    # parser.add_argument('--use_feature_moe', action='store_true', default=False)
    

    args = parser.parse_args()
    #ncn/ncnc:
    args.predp = args.gnndp = args.dropout
    args.mplayers = args.num_layers
    args.nnlayers  = args.num_layers_predictor

 ##buddy
    args.label_dropout = args.feature_dropout = args.dropout
    args = check_parameter(args)
    # args = generate_prefix(args)
    print(args)
    return args

def check_parameter(args):
    # add checking the validty of the args, for example, the number of split
    if (args.max_hash_hops == 1) and (not args.use_zero_one):
        print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
    if args.data_name == 'ogbl-ddi':
        args.use_feature = 0  # dataset has no features
        assert args.sign_k > 0, '--sign_k must be set to > 0 i.e. 1,2 or 3 for ogbl-ddi'
    
 
    return args


    
def generate_prefix(args):
    # add some prefix for easy saving
    prefix_dict = {}

    prefix_dict["model"] = f"{args.algo_name}"
    # prefix_dict["model"] += f"{args.hidden_dimension}_{args.num_layers}"
    prefix_dict["train"] = f"{args.random_seed}"

    prefix_dict["dataset"] = f"{args.dataset}"
    
    prefix_dict["tuning"] = f""

    args.prefixs = prefix_dict
    
    return args



def str2bool(x):
    """
    hack to allow wandb to tune boolean cmd args
    :param x: str of bool
    :return: bool
    """
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

    
