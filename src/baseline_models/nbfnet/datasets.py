import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.data.collate import collate
from ogb.linkproppred import PygLinkPropPredDataset

class IndRelLinkPredDataset(InMemoryDataset):

    urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
        ]
    }

    def __init__(self, root, name, version, transform=None, pre_transform=None):
        self.name = name
        self.version = version
        assert name in ["FB15k-237", "WN18RR"]
        assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name



def build_ogb_dataset(dataset_name):

    dataset = PygLinkPropPredDataset(name=dataset_name)
    data = dataset[0]
    edge_index = data.edge_index
   
    num_nodes = data.num_nodes
    num_relations = 1

    split_edge = dataset.get_edge_split()
    
    if dataset_name != 'ogbl-citation2':
        pos_train_edge = torch.transpose(split_edge['train']['edge'],1,0)

        pos_valid_edge = torch.transpose(split_edge['valid']['edge'],1,0)
        neg_valid_edge = torch.transpose(split_edge['valid']['edge_neg'], 1, 0)
        pos_test_edge = torch.transpose(split_edge['test']['edge'], 1, 0)
        neg_test_edge = torch.transpose(split_edge['test']['edge_neg'], 1, 0)

    edge_type_train = torch.zeros((pos_train_edge.size(1))).long()  # Uni-relational...all 0s
    edge_type_valid = torch.zeros((pos_valid_edge.size(1))).long() 
    edge_type_test = torch.zeros((pos_test_edge.size(1))).long() 

    edge_type_valid_neg = torch.zeros((neg_valid_edge.size(1))).long() 
    edge_type_test_neg = torch.zeros((neg_test_edge.size(1))).long() 

    edge_type_with_inv = torch.cat([edge_type_train, edge_type_train + num_relations])


    train_data = Data(edge_index=edge_index, edge_type=edge_type_with_inv, num_nodes=num_nodes,
                        target_edge_index=pos_train_edge, target_edge_type=edge_type_train, target_neg=neg_valid_edge )

    valid_data = Data(edge_index=edge_index, edge_type=edge_type_with_inv, num_nodes=num_nodes,
                        target_edge_index=pos_valid_edge, target_edge_type=edge_type_valid, target_neg=neg_valid_edge, target_neg_type=edge_type_valid_neg)
        
    test_data = Data(edge_index=edge_index, edge_type=edge_type_with_inv, num_nodes=num_nodes,
                        target_edge_index=pos_test_edge, target_edge_type=edge_type_test, target_neg=neg_test_edge, target_neg_type=edge_type_test_neg)
    
    dataset = InMemoryDataset()
    # dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    
    dataset.num_relations = num_relations 
    
    return train_data,valid_data, test_data, dataset

def build_citation_dataset(dataset_name, data):
    """
    Build either "Cora", "Pubmed", or "Citeseer".
    """
    # dataset = Planetoid('.', name=dataset_name)

    num_relations = 1
    num_nodes = data['x'].size(0)
    # num_edges = dataset[0].edge_index.shape[-1]

    train_edge_index = torch.transpose(data['train_pos'],1, 0)
    valid_pos =  torch.transpose(data['valid_pos'], 1, 0)
    valid_neg =  torch.transpose(data['valid_neg'], 1, 0)

    test_pos =  torch.transpose(data['test_pos'], 1, 0)
    test_neg =  torch.transpose(data['test_neg'], 1, 0)
    

    edge_type_train = torch.zeros((train_edge_index.size(1))).long()  # Uni-relational...all 0s
    edge_type_valid = torch.zeros((valid_pos.size(1))).long() 
    edge_type_test = torch.zeros((test_pos.size(1))).long() 

    # Shuffles all indices and split to create raw split mask
    # index = torch.arange(edge_index.shape[1])
    # shuffled_index = index[torch.randperm(index.shape[0])]

    # Split edges into train/valid/test indices
    # Already shuffled so can just split sequentially
    # num_edges = shuffled_index.shape[0]
    # train_ix = shuffled_index[: int(num_edges * splits[0])]
    # valid_ix = shuffled_index[int(num_edges * splits[0]) : int(num_edges * splits[0]) + int(num_edges * splits[1])]
    # test_ix = shuffled_index[int(num_edges * splits[0]) + int(num_edges * splits[1]) : ]

    # convert to a mask format
    # train_mask = index_to_mask(train_ix, num_edges)
    # valid_mask = index_to_mask(valid_ix, num_edges)
    # test_mask = index_to_mask(test_ix, num_edges)
    
    # Only train is used for propagation
    # So we only add inverse to them
    edge_index_with_inv = torch.cat([train_edge_index,train_edge_index.flip(0)], dim=-1)
    edge_type_with_inv = torch.cat([edge_type_train, edge_type_train + num_relations])

    train_data = Data(edge_index=edge_index_with_inv, edge_type=edge_type_with_inv, num_nodes=num_nodes,
                        target_edge_index=train_edge_index, target_edge_type=edge_type_train, target_neg=test_neg)
    valid_data = Data(edge_index=edge_index_with_inv, edge_type=edge_type_with_inv, num_nodes=num_nodes,
                        target_edge_index=valid_pos, target_edge_type=edge_type_valid, target_neg=valid_neg)
    test_data = Data(edge_index=edge_index_with_inv, edge_type=edge_type_with_inv, num_nodes=num_nodes,
                        target_edge_index=test_pos, target_edge_type=edge_type_test, target_neg=test_neg)
    
    dataset = InMemoryDataset()
    # dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    
    dataset.num_relations = num_relations * 2
    
    return train_data,valid_data, test_data, dataset




def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask