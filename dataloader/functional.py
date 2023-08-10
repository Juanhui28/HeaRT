import numpy as np
from torch_geometric.data import Data, InMemoryDataset

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def make_train_eval_data(args, train_dataset, num_nodes, n_pos_samples=5000, negs_per_pos=1000):
    """
    A much smaller subset of the training data to get a comparable (with test and val) measure of training performance
    to diagnose overfitting
    @param args: Namespace object of cmd args
    @param train_dataset: pyG Dataset object
    @param n_pos_samples: The number of positive samples to evaluate the training set on
    @return: HashedTrainEvalDataset
    """
    # ideally the negatives and the subgraph features are cached and just read from disk
    # need to save train_eval_negs_5000 and train_eval_subgraph_features_5000 files
    # and ensure that the order is always the same just as with the other datasets
    print('constructing dataset to evaluate training performance')
    dataset_name = args.dataset_name
    pos_sample = train_dataset.pos_edges[:n_pos_samples]  # [num_edges, 2]
    negs_name = f'{ROOT_DIR}/dataset/{dataset_name}/train_eval_negative_samples_{negs_per_pos}.pt'
    print(f'looking for negative edges at {negs_name}')
    if os.path.exists(negs_name):
        print('loading negatives from disk')
        neg_sample = torch.load(negs_name)
    else:
        print('negatives not found on disk. Generating negatives')
        neg_sample = get_same_source_negs(num_nodes, negs_per_pos, pos_sample.t()).t()  # [num_neg_edges, 2]
        torch.save(neg_sample, negs_name)
    # make sure these are the correct negative samples with source nodes corresponding to the positive samples
    assert torch.all(torch.eq(pos_sample[:, 0].repeat_interleave(negs_per_pos), neg_sample[:,
                                                                                0])), 'negatives have different source nodes to positives. Delete train_eval_negative_samples_* and subgraph features and regenerate'
    links = torch.cat([pos_sample, neg_sample], 0)  # [n_edges, 2]
    labels = [1] * pos_sample.size(0) + [0] * neg_sample.size(0)
    # if train_dataset.use_RA:
    #     pos_RA = train_dataset.RA[:n_pos_samples]
    #     neg_RA = RA(train_dataset.A, neg_sample, batch_size=2000000)[0]
    #     RA_links = torch.cat([pos_RA, neg_RA], dim=0)
    # else:
    #     RA_links = None
    pos_sf = train_dataset.subgraph_features[:n_pos_samples]
    # try to read negative subgraph features from disk or generate them
    subgraph_cache_name = f'{ROOT_DIR}/dataset/{dataset_name}/train_eval_negative_samples_{negs_per_pos}_subgraph_featurecache.pt'
    print(f'looking for subgraph features at {subgraph_cache_name}')
    if os.path.exists(subgraph_cache_name):
        neg_sf = torch.load(subgraph_cache_name).to(pos_sf.device)
        print(f"cached subgraph features found at: {subgraph_cache_name}")
        assert neg_sf.shape[0] == len(
            neg_sample * negs_per_pos), 'subgraph features are a different shape link object. Delete subgraph features file and regenerate'
    else:  # generate negative subgraph features
        #  we're going to need the hashes
        file_stub = dataset_name.replace('-', '_')  # pyg likes to add -
        if args.max_hash_hops == 3:
            hash_name = f'{ROOT_DIR}/dataset/{dataset_name}/{file_stub}_elph__train_3hop_hashcache.pt'
        else:
            hash_name = f'{ROOT_DIR}/dataset/{dataset_name}/{file_stub}_elph__train_hashcache.pt'
        print(f'looking for hashes at {hash_name}')
        eh = ElphHashes(args)
        if os.path.exists(hash_name):
            hashes = torch.load(hash_name)
            print(f"cached hashes found at: {hash_name}")
        else:  # need to generate the hashes, but this is a corner case as they should have been generated to make the training dataset
            hashes, cards = eh.build_hash_tables(num_nodes, train_dataset.edge_index)
            torch.save(hashes, hash_name)
        print('caching subgraph features for negative samples to evaluate training performance')
        neg_sf = eh.get_subgraph_features(neg_sample, hashes, cards)
        torch.save(neg_sf, subgraph_cache_name)
    subgraph_features = torch.cat([pos_sf, neg_sf], dim=0)
    train_eval_dataset = HashedTrainEvalDataset(links, labels, subgraph_features, RA_links, train_dataset)
    return train_eval_dataset


def get_pos_neg_edges(data, sample_frac=1):
    """
    extract the positive and negative supervision edges (as opposed to message passing edges) from data that has been
     transformed by RandomLinkSplit
    :param data: A train, val or test split returned by RandomLinkSplit
    :return: positive edge_index, negative edge_index.
    """
    device = data.edge_index.device
    edge_index = data['edge_label_index'].to(device)
    labels = data['edge_label'].to(device)
    pos_edges = edge_index[:, labels == 1].t()
    neg_edges = edge_index[:, labels == 0].t()
    if sample_frac != 1:
        n_pos = pos_edges.shape[0]
        np.random.seed(123)
        perm = np.random.permutation(n_pos)
        perm = perm[:int(sample_frac * n_pos)]
        pos_edges = pos_edges[perm, :]
        neg_edges = neg_edges[perm, :]
    return pos_edges.to(device), neg_edges.to(device)


def get_same_source_negs(num_nodes, num_negs_per_pos, pos_edge):
    """
    The ogb-citation datasets uses negatives with the same src, but different dst to the positives
    :param num_nodes: Int node count
    :param num_negs_per_pos: Int
    :param pos_edge: Int Tensor[2, edges]
    :return: Int Tensor[2, edges]
    """
    print(f'generating {num_negs_per_pos} single source negatives for each positive source node')
    dst_neg = torch.randint(0, num_nodes, (1, pos_edge.size(1) * num_negs_per_pos), dtype=torch.long)
    src_neg = pos_edge[0].repeat_interleave(num_negs_per_pos)
    return torch.cat([src_neg.unsqueeze(0), dst_neg], dim=0)


def neighbors(fringe, A, outgoing=True):
    """
    Retrieve neighbours of nodes within the fringe
    :param fringe: set of node IDs
    :param A: scipy CSR sparse adjacency matrix
    :param outgoing: bool
    :return:
    """
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def get_src_dst_degree(src, dst, A, max_nodes):
    """
    Assumes undirected, unweighted graph
    :param src: Int Tensor[edges]
    :param dst: Int Tensor[edges]
    :param A: scipy CSR adjacency matrix
    :param max_nodes: cap on max node degree
    :return:
    """
    src_degree = A[src].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    dst_degree = A[dst].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    return src_degree, dst_degree
