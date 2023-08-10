class SEALDynamicDataset(LPDataset):
    def __init__(self, root, data, pos_edges, neg_edges, num_hops, percent=1., use_coalesce=False, node_label='drnl',
                 ratio_per_hop=1.0, max_nodes_per_hop=None, max_dist=1000, directed=False, sign=False, k=None,
                 **kwargs):
        self.num_hops = num_hops
        self.percent = percent
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.sign = sign
        self.k = k
        super(SEALDynamicDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0).tolist()
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None
        
        # TODO: add the new embedding that required in the dataset

    def len(self):
        return len(self.links)

    # return subgraph
    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, self.max_nodes_per_hop)
        if self.sign:
            x = [self.data.x]
            x += [self.data[f'x{i}'] for i in range(1, self.k + 1)]
        else:
            x = self.data.x
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=x,
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = self.construct_pyg_graph(*tmp, self.node_label, self.max_dist, src_degree, dst_degree)

        return data, label
    
    def construct_pyg_graph(self, node_ids, adj, dists, node_features, y, node_label='drnl', max_dist=1000, src_degree=None,
                        dst_degree=None):
        """
        Constructs a pyg graph for this subgraph and adds an attribute z containing the node_label
        @param node_ids: list of node IDs in the subgraph
        @param adj: scipy sparse CSR adjacency matrix
        @param dists: an n_nodes list containing shortest distance (in hops) to the src or dst node
        @param node_features: The input node features corresponding to nodes in node_ids
        @param y: scalar, 1 if positive edges, 0 if negative edges
        @param node_label: method to add the z attribute to nodes
        @return:
        """
        u, v, r = ssp.find(adj)
        num_nodes = adj.shape[0]

        node_ids = torch.LongTensor(node_ids)
        u, v = torch.LongTensor(u), torch.LongTensor(v)
        r = torch.LongTensor(r)
        edge_index = torch.stack([u, v], 0)
        edge_weight = r.to(torch.float)
        y = torch.tensor([y])
        if node_label == 'drnl':  # DRNL
            z = drnl_node_labeling(adj, 0, 1, max_dist)
        elif node_label == 'hop':  # mininum distance to src and dst
            z = torch.tensor(dists)
        elif node_label == 'zo':  # zero-one labeling trick
            z = (torch.tensor(dists) == 0).to(torch.long)
        elif node_label == 'de':  # distance encoding
            z = de_node_labeling(adj, 0, 1, max_dist)
        elif node_label == 'de+':
            z = de_plus_node_labeling(adj, 0, 1, max_dist)
        elif node_label == 'degree':  # this is technically not a valid labeling trick
            z = torch.tensor(adj.sum(axis=0)).squeeze(0)
            z[z > 100] = 100  # limit the maximum label to 100
        else:
            z = torch.zeros(len(dists), dtype=torch.long)
        data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                    node_id=node_ids, num_nodes=num_nodes, src_degree=src_degree, dst_degree=dst_degree)
    return data


    # add the fix loader that we need




# TODO: do not use the inmemery dataset but just load 10000 data once.It could be fast
# TODO: maybe we can merge it with the dynamic one
# The original implementation use the torch_geometric loader, here we replace it with the original version
class SEALDataset(LPDataset):
    def __init__(self, root, data, pos_edges, neg_edges, num_hops, percent=1., split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, max_dist=1000, directed=False, sign=False, k=None):
        super(LPDataset, self).__init__(root, split, data, pos_edges, neg_edges, use_feature, use_coalesce, directed)
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent 
        # For large dataselt, SEAL only used a small ratio edges for train. 
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.sign = sign
        self.k = k
        self.data, self.slices = torch.load(self.processed_paths[0])

    # add it for it could be very slow for GNN
    # TODO: rewrite one, please.
    @property
    def processed_file_names(self):
        if self.percent == 1.:
            name = f'SEAL_{self.split}_data'
        else:
            name = f'SEAL_{self.split}_data_{self.percent}'
        name += '.pt'
        return [name]

    def process(self):
        if self.use_coalesce:  # compress multi-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            self.pos_edges, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.max_dist, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(
            self.neg_edges, A, self.data.x, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.max_dist, self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list






def drnl_hash_function(dist2src, dist2dst):
    """
    mapping from source and destination distances to a single node label e.g. (1,1)->2, (1,2)->3
    @param dist2src: Int Tensor[edges] shortest graph distance to source node
    @param dist2dst: Int Tensor[edges] shortest graph distance to source node
    @return: Int Tensor[edges] of labels
    """
    dist = dist2src + dist2dst

    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    # the src and dst nodes always get a score of 1
    z[dist2src == 0] = 1
    z[dist2dst == 0] = 1
    return z


def get_drnl_lookup(max_dist, num_hops):
    """
    A lookup table from DRNL labels to index into a contiguous tensor. DRNL labels are not contiguous and this
    lookup table is used to index embedded labels
    """
    max_label = get_max_label('drnl', max_dist, num_hops)
    res_arr = [None] * (max_label + 1)
    res_arr[1] = (1, 0)
    for src in range(1, num_hops + 1):
        for dst in range(1, max_dist + 1):
            label = drnl_hash_function(torch.tensor([src]), torch.tensor([dst]))
            res_arr[label] = (src, dst)
    z_to_idx = {}
    idx_to_dst = {}
    counter = 0
    for idx, elem in enumerate(res_arr):
        if elem is not None:
            z_to_idx[idx] = counter
            idx_to_dst[counter] = (elem)
            counter += 1
    return z_to_idx, idx_to_dst


def get_max_label(method, max_dist, num_hops):
    if method in {'de', 'de+'}:
        max_label = max_dist
    elif method in {'drnl-', 'drnl'}:
        max_label = drnl_hash_function(torch.tensor([num_hops]), torch.tensor([max_dist])).item()
    else:
        raise NotImplementedError
    return max_label


def drnl_node_labeling(adj, src, dst, max_dist=100):
    """
    The heuristic proposed in "Link prediction based on graph neural networks". It is an integer value giving the 'distance'
    to the (src,dst) edge such that src = dst = 1, neighours of dst,src = 2 etc. It implements
    z = 1 + min(d_x, d_y) + (d//2)[d//2 + d%2 - 1] where d = d_x + d_y
    z is treated as a node label downstream. Even though the labels measures of distance from the central edge, they are treated as
    categorical objects and embedded in an embedding table of size max_z * hidden_dim
    @param adj:
    @param src:
    @param dst:
    @return:
    """
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)
    dist2src[dist2src > max_dist] = max_dist

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)
    dist2dst[dist2dst > max_dist] = max_dist

    z = drnl_hash_function(dist2src, dist2dst)
    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 1, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 1, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist

    return dist.to(torch.long)