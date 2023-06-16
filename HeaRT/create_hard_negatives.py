"""
Create hard negative samples for validation and testing evaluation
"""
import os 
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata
from argparse import ArgumentParser 
from sklearn.metrics.pairwise import cosine_similarity

from hard_utils import *
from calc_ppr import create_ppr_matrices


ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


def save_samples(samples, file_name):
    print("Saving samples...")
    with open(file_name, "wb") as f:
        np.save(f, samples)


def calc_PPR_scores(args):
    """
    Calc the PPR for nodes using the Anderson algorithm.

    Read in matrices if they already exist
    """
    dataset_dir = os.path.join(ROOT_DIR, "data", "ppr", args.dataset)
    base_ppr_file = os.path.join(dataset_dir, f"sparse_adj-{str(args.alpha).replace('.', '')}_eps-{str(args.eps).replace('.', '')}.pt")

    # If they don't exist we create them first 
    if not os.path.isfile(base_ppr_file):
        print("PPR matrices don't exist")
        create_ppr_matrices(args.dataset, args.alpha, args.eps, args.use_val_in_test)
    else:
        print("PPR matrices already exist. Loading...(this may take a minute)")

    base_ppr = torch.load(base_ppr_file) 

    if args.use_val_in_test:
        test_ppr_file = os.path.join(dataset_dir, f"sparse_adj-{str(args.alpha).replace('.', '')}_eps-{str(args.eps).replace('.', '')}_val.pt")
        test_ppr = torch.load(test_ppr_file)    
    else:
        test_ppr = base_ppr
    
    return base_ppr, test_ppr
        

def calc_CN_metric(data, metric="RA", use_val=False):
    """
    Calc CN/RA for all node pairs
    """
    print(f"Calculating {metric}...")

    if use_val:
        adj = data['train_valid_adj']
    else:
        adj = data['adj_t']

    # Convert adj to non-weighted for collab
    if "collab" in data['dataset'].lower():
        non_weighted_adj = adj.set_value((adj.storage.value() > 0).float(), layout="coo")
    else:
        non_weighted_adj = adj

    if metric == "RA":
        # Weight adjacency by 1 / edge-weighted degree
        reciprocal_degree = 1 / adj.sum(dim=0).to_dense().unsqueeze(0)  
        reciprocal_degree = torch.nan_to_num(reciprocal_degree)

        weighted_adj = non_weighted_adj * reciprocal_degree
        cn_scores = non_weighted_adj @ weighted_adj
    else:
        cn_scores = non_weighted_adj @ non_weighted_adj

    return cn_scores


def calc_feat_sim(data):
    """
    Calculate the feature similarity of all node pairs.

    We use cosine similarity

    NOTE: Don't attempt this on OGB
    """
    print("Calculating Feature Similarity...")
    return cosine_similarity(data['x'].numpy(), data['x'].numpy())


def rank_score_matrix(row):
    """
    Rank from largest->smallest
    """
    num_greater_zero = (row > 0).sum().item()

    # Ignore 0s and -1s in ranking
    # Note: default is smallest-> largest so reverse
    if num_greater_zero > 0:
        ranks_row = rankdata(row[row > 0], method='min')
        ranks_row = ranks_row.max() - ranks_row + 1
        max_rank = ranks_row.max()
    else:
        ranks_row = []
        max_rank = 0

    # Overwrite row with ranks
    # Also overwrite 0s with max+1 and -1s with max+2
    row[row > 0] = ranks_row
    row[row == 0] = max_rank + 1
    row[row < 0] = max_rank + 2

    return row


def rank_and_merge_node(node_cn_scores, node_ppr_scores, node_feat_scores, true_pos_mask, data, args):
    """
    Do so for a single node
    """
    pnorm = args.pnorm
    k = args.num_samples // 2 
    agg_func = np.mean if args.agg == "mean" else np.min

    if node_feat_scores is not None:
        node_feat_scores[true_pos_mask] = -1  # TODO: Is this already true?

        # Nodes that are 0 for all scores. Needed later when selecting top K
        zero_nodes_score_mask = ((node_cn_scores == 0) & (node_ppr_scores == 0) & (node_feat_scores == 0)).numpy()
    else:
        # Nodes that are 0 for all scores. Needed later when selecting top K
        zero_nodes_score_mask = ((node_cn_scores == 0) & (node_ppr_scores == 0)).numpy()

    # Individual ranks + combine
    node_cn_ranks = rank_score_matrix(node_cn_scores.numpy())
    node_ppr_ranks = rank_score_matrix(node_ppr_scores.numpy())

    if node_feat_scores is not None:
        node_feat_ranks = rank_score_matrix(node_feat_scores)
        combined_node_ranks = agg_func([node_cn_ranks**pnorm, node_ppr_ranks**pnorm, node_feat_ranks**pnorm], axis=0)
    else:
        combined_node_ranks = agg_func([node_cn_ranks**pnorm, node_ppr_ranks**pnorm], axis=0)

    # If enough non-zero scores we use just take top-k
    # Otherwise we have to randomly select from 0 scores        
    max_greater_zero = data['num_nodes'] - zero_nodes_score_mask.sum().item() - true_pos_mask.sum().item()

    # NOTE: Negate when using torch.topk since 1=highest
    if max_greater_zero >= k:
        node_topk = torch.topk(torch.from_numpy(-combined_node_ranks), k).indices
        node_topk = node_topk.numpy()
    else:
        # First just take whatever non-zeros there are
        # Then choose random
        node_greater_zero = torch.topk(torch.from_numpy(-combined_node_ranks), max_greater_zero).indices
        node_greater_zero = node_greater_zero.numpy()

        node_zero_score_ids = zero_nodes_score_mask.nonzero()[0]
        node_zero_rand = np.random.choice(node_zero_score_ids, k-max_greater_zero)

        node_topk = np.concatenate((node_greater_zero, node_zero_rand))

    return node_topk.reshape(-1, 1)


def rank_and_merge_edges(edges, cn_scores, ppr_scores, feat_sim_scores, data, args, test=False):
    """
    For each edge we get the rank for the types of scores for each node and merge them together to one rank

    Using that we get the nodes with the top k ranks
    """
    all_topk_edges = []
    k = args.num_samples // 2 

    # Used to determine positive samples to filter
    # For testing we also include val samples in addition to train
    if test:
        adj = data['train_valid_adj']
    else:
        adj = data['adj_t']

    
    for edge in tqdm(edges, "Ranking Scores"):
        source, target = edge[0].item(), edge[1].item()

        source_adj = adj[source].to_dense().squeeze(0).bool()
        source_cn_scores = cn_scores[source].to_dense().squeeze(0)
        source_ppr_scores = ppr_scores[source].to_dense().squeeze(0)

        target_adj = adj[target].to_dense().squeeze(0).bool()
        target_cn_scores = cn_scores[target].to_dense().squeeze(0)
        target_ppr_scores = ppr_scores[target].to_dense().squeeze(0)

        # Filter self-loops and positive samples by setting to -1
        # Also filter edge we want to predict
        source_true_pos_mask = source_adj
        source_true_pos_mask[source], source_true_pos_mask[target] = 1, 1
        source_cn_scores[source_true_pos_mask], source_ppr_scores[source_true_pos_mask] = -1, -1 

        target_true_pos_mask = target_adj
        target_true_pos_mask[target], target_true_pos_mask[source] = 1, 1
        target_cn_scores[target_true_pos_mask], target_ppr_scores[target_true_pos_mask] =  -1, -1 

        if feat_sim_scores is not None:
            source_feat_scores = feat_sim_scores[source]
            target_feat_scores = feat_sim_scores[target]
            source_feat_scores[source_true_pos_mask] = -1
            target_feat_scores[target_true_pos_mask] = -1
        else:
            source_feat_scores, target_feat_scores = None, None

        source_topk_nodes = rank_and_merge_node(source_cn_scores, source_ppr_scores, source_feat_scores, source_true_pos_mask, data, args)
        source_topk_edges = np.concatenate((np.repeat(source, k).reshape(-1, 1), source_topk_nodes), axis=-1)

        target_topk_nodes = rank_and_merge_node(target_cn_scores, target_ppr_scores, target_feat_scores, target_true_pos_mask, data, args)
        target_topk_edges = np.concatenate((target_topk_nodes, np.repeat(target, k).reshape(-1, 1)), axis=-1)
        
        edge_samples = np.concatenate((source_topk_edges, target_topk_edges))
        all_topk_edges.append(edge_samples)

    return np.stack(all_topk_edges)


def calc_all_heuristics(args):
    """
    Calc and store top-k negative samples for each sample

    Uses feature similarity for non-OGB datasets
    """
    dataset_dir = os.path.join(ROOT_DIR, "data", "hard_negative", args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    if "ogb" in args.dataset.lower():
        data = get_data_ogb(args)
        feat_sim_scores = None
    else:
        data = get_data_planetoid(args.dataset)
        feat_sim_scores = calc_feat_sim(data)

    val_cn_scores  = calc_CN_metric(data, args.cn_metric)
    test_cn_scores = calc_CN_metric(data, args.cn_metric, True) if args.use_val_in_test else val_cn_scores
    val_ppr_scores, test_ppr_scores = calc_PPR_scores(args)

    print("\n>>> Valid")
    val_neg_samples = rank_and_merge_edges(data['valid_pos'], val_cn_scores, val_ppr_scores, feat_sim_scores, data, args)
    save_samples(val_neg_samples, os.path.join(dataset_dir, f"valid_samples_agg-{args.agg}_norm-{args.pnorm}.npy"))

    print("\n>>> Test")
    test_neg_samples = rank_and_merge_edges(data['test_pos'], test_cn_scores, test_ppr_scores, feat_sim_scores, data, args, test=True)
    save_samples(test_neg_samples, os.path.join(dataset_dir, f"test_samples_agg-{args.agg}_norm-{args.pnorm}.npy"))



def main():
    parser = ArgumentParser(description="Create hard negative samples")
    parser.add_argument("--dataset", help="Dataset to create samples for", type=str, required=True)
    parser.add_argument("--use-val-in-test", action='store_true', default=False)

    parser.add_argument("--cn-metric", help="Either 'RA' or 'CN'", type=str, default="RA")
    parser.add_argument("--pnorm", help="P-norm when combining ranks", type=int, default=1)
    parser.add_argument("--agg", help="For combining ranks. Either 'mean' or 'min'", type=str, default="mean")
    parser.add_argument("--num-samples", help="Number of negative samples per sample", type=int, default=500)

    # For PPR
    parser.add_argument("--eps", help="Stopping criterion threshold", type=float, default=5e-5)
    parser.add_argument("--alpha", help="Teleportation probability", type=float, default=0.15)

    args = parser.parse_args()

    set_seeds()
    calc_all_heuristics(args)


if __name__ == "__main__":
    main()
