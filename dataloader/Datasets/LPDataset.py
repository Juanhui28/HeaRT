"""
constructing the hashed data objects used by elph and buddy
"""

import os
from time import time
from typing import Callable, Optional
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce
import scipy.sparse as ssp
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import InMemoryDataset
from torch_sparse import SparseTensor
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
from .dataprocess import filter_by_year, get_ogb_data, wrap_data

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import networkx as nx
import dgl
# from model.PEG.Graph_embedding import DeepWalk
# from model.PEG.utils import laplacian_positional_encoding
# from src.heuristics import RA
# from src.utils import ROOT_DIR, get_src_dst_degree, get_pos_neg_edges, get_same_source_negs
# from src.hashing import ElphHashes


class LPDataset(Dataset):

    def __init__(self, args):

        super(LPDataset, self).__init__()

        self.dir_path =  os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
        self.data_name = args.data_name
        self.args = args
        self.save_data_name = self.data_name.replace('-', '_')
        if self.data_name == 'cora' or self.data_name == 'citeseer' or self.data_name == 'pubmed':
            self.read_smalldata()
            self.directed = False
            self.max_x = -1
        else:
            self.read_ogb()
            self.max_x = -1
            if  self.data_name == 'ogbl-ppa' :
                self.ncnx = torch.argmax(self.x, dim=-1)
                self.max_x = torch.max(self.ncnx).item()
        
        ## for negonn which considers multi-hop cn
        self.A2, self.full_A2 = self.process_A2()

        self._num_features = self.x.size(1)

    def read_smalldata(self):
      
        node_set = set()
        train_pos, valid_pos, test_pos = [], [], []
        train_neg, valid_neg, test_neg = [], [], []

        for split in ['train', 'test', 'valid']:

            path = self.dir_path+'/dataset' + '/{}/{}_pos.txt'.format(self.data_name, split)

            for line in open(path, 'r'):
                sub, obj = line.strip().split('\t')
                sub, obj = int(sub), int(obj)
                
                node_set.add(sub)
                node_set.add(obj)
                
                if sub == obj:
                    continue

                if split == 'train': 
                    train_pos.append((sub, obj))
                    

                if split == 'valid': valid_pos.append((sub, obj))  
                if split == 'test': test_pos.append((sub, obj))
        
        num_nodes = len(node_set)
        print('the number of nodes in ' + self.data_name + ' is: ', num_nodes)

        for split in ['train','test', 'valid']:

            path = self.dir_path+'/dataset' + '/{}/{}_neg.txt'.format(self.data_name, split)

           
            for line in open(path, 'r'):
                sub, obj = line.strip().split('\t')
                sub, obj = int(sub), int(obj)
                # if sub == obj:
                #     continue
                
                if split == 'train': 
                    train_neg.append((sub, obj))

                if split == 'valid': 
                    valid_neg.append((sub, obj))
                
                if split == 'test': 
                    test_neg.append((sub, obj))

       
        edge_index = to_undirected(torch.tensor(train_pos).t())

        edge_weight = torch.ones(edge_index.size(1), dtype=float)
        
        # A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

        adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
       

        train_pos_tensor = torch.tensor(train_pos)
        train_neg = torch.tensor(train_neg)

        valid_pos = torch.tensor(valid_pos)
        valid_neg =  torch.tensor(valid_neg)

        test_pos =  torch.tensor(test_pos)
        test_neg =  torch.tensor(test_neg)

        idx = torch.randperm(train_pos_tensor.size(0))
        self.idx = idx[:valid_pos.size(0)]
        train_val = train_pos_tensor[self.idx]


        feature_embeddings = torch.load(self.dir_path+'/dataset' + '/{}/{}'.format(self.data_name, 'gnn_feature'))
        feature_embeddings = feature_embeddings['entity_embedding']

        self.split_edge = {'train': {}, 'valid': {}, 'test': {}, 'train_val':{}}
        self.split_edge['train']['edge'] = train_pos_tensor
        self.split_edge['train']['edge_neg'] = train_neg

        self.split_edge['train_val']['edge'] = train_val
        self.split_edge['train_val']['edge_neg'] = valid_neg

        self.split_edge['valid']['edge']= valid_pos
        self.split_edge['valid']['edge_neg'] = valid_neg
        self.split_edge['test']['edge']  = test_pos
        self.split_edge['test']['edge_neg']  = test_neg
        
        self.x = feature_embeddings
        self.adj = adj
        self.num_nodes = num_nodes
        self.edge_weight = edge_weight
        self.edge_index = edge_index
        path = os.path.join(self.dir_path, 'dataset', self.args.data_name, self.args.data_name+'-n2v-embedding.pt')
        self.n2vfeat = torch.load(path)
        
        self.splits, self.full_edge_index, self.full_edge_weight,  self.full_adj_t  = wrap_data(self.x, self.edge_index, self.edge_weight, self.split_edge, self.args, self.num_nodes)
        if self.full_adj_t  != None:
            self.full_adj_t = self.full_adj_t.to(torch.float)
        

    def read_ogb(self):

        self.directed = False
        dataset = PygLinkPropPredDataset(name=self.data_name)
        
        if self.data_name == 'ogbl-ddi':
            dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
            dataset.data.edge_weight = torch.ones(dataset.data.edge_index.size(1), dtype=int)
        
        if 'citation' in self.data_name:
            self.directed = True
        
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        if self.data_name == 'ogbl-collab' and self.args.year > 0:  # filter out training edges before args.year
            data, split_edge = filter_by_year(data, split_edge, self.args.year)
            
        self.split_edge, self.idx = get_ogb_data(data, split_edge, self.data_name, self.args.use_train_val)
        self.x = data.x
        self.num_nodes = data.num_nodes
        self.edge_weight  = torch.ones(data.edge_index.size(1), dtype=float)
        self.edge_index = data.edge_index
        if hasattr(data, 'edge_weight'):
            if data.edge_weight != None:
                self.edge_weight = data.edge_weight

        self.splits, self.full_edge_index, self.full_edge_weight, self.full_adj_t = wrap_data(self.x, self.edge_index, self.edge_weight, self.split_edge, self.args, self.num_nodes, data)
        if self.full_adj_t  != None:
            self.full_adj_t = self.full_adj_t.to(torch.float)
        data = T.ToSparseTensor()(data)

        # import ipdb
        # ipdb.set_trace()
        self.adj = SparseTensor.from_edge_index(self.edge_index, self.edge_weight.view(-1), [self.num_nodes, self.num_nodes]).to(torch.float)
        self.process_adj(self.args)
        path = os.path.join(self.dir_path, 'dataset', self.save_data_name, self.args.data_name+'-n2v-embedding.pt')
        self.n2vfeat = torch.load(path)
        
        


    def process_adj(self, args):
        
        if args.data_name == 'ogbl-citation2': 
            self.adj_t = self.adj_t.to_symmetric()
            if args.gnn_model == 'GCN':
                adj_t = self.adj_t.set_diag()
                deg = adj_t.sum(dim=1).to(torch.float)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
                self.adj_t = adj_t



    def process_A2(self):
        
        A = ssp.csr_matrix((self.edge_weight.reshape(-1), (self.edge_index[0], self.edge_index[1])), 
                       shape=(self.num_nodes, self.num_nodes))
       
        if  self.data_name == 'cora' or self.data_name == 'citeseer' or self.data_name == 'pubmed' or self.data_name == 'ogbl-collab':
            
            A2 = A * A
            A = A + self.args.beta*A2
      
        if self.args.use_valedges_as_input:
            full_A = ssp.csr_matrix((self.full_edge_weight.reshape(-1), (self.full_edge_index[0], self.full_edge_index[1])), 
                       shape=(self.num_nodes, self.num_nodes))
        
            A2 = full_A * full_A
            full_A = full_A + self.args.beta*A2
        else:
            full_A = None
        # import ipdb
        # ipdb.set_trace()
        return A, full_A
    
    def process_PEG_data(self):
        if self.data_name == 'cora' or self.data_name == 'citeseer' or self.data_name == 'pubmed':
            pca=PCA(n_components=self.args.hidden_channels)
            features=pca.fit_transform(np.array(self.x))
            features = torch.tensor(features)
            peg_pca_features = features.type(torch.FloatTensor)
        else:
            peg_pca_features = self.x
        
        if self.args.PE_method == 'DW':
            path = self.dir_path + 'dataset/'+self.save_data_name+'/dw_emb'
            if os.path.exists():
                embeddings = torch.load(path, map_location=torch.device('cpu'))

            else:

                G = nx.from_edgelist(np.array(self.edge_index).T)
                model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
                model_emb.train(embed_size = self.args.PE_dim)# train model
                emb = model_emb.get_embeddings()# get embedding vectors
                embeddings = []
                for i in range(len(emb)):
                    embeddings.append(emb[i])
                embeddings = torch.tensor(np.array(embeddings))
                embeddings  = embeddings
                torch.save(embeddings, path)

        elif self.args.PE_method == 'LE':  
            path = self.dir_path + 'dataset/'+self.save_data_name+'/le_emb'
            if os.path.exists(path):
                embeddings  = torch.load(path,  map_location=torch.device('cpu'))
            else:
                G = nx.from_edgelist(np.array(self.edge_index).T)
                G = nx.to_scipy_sparse_matrix(G)
                g = dgl.from_scipy(G)
                embeddings = laplacian_positional_encoding(g, self.args.PE_dim)

                if self.data_name == 'cora' or self.data_name == 'citeseer' or self.data_name == 'pubmed':
                    embeddings = normalize(np.array(embeddings), norm='l2', axis=1, copy=True, return_norm=False)
               
                embeddings = torch.tensor(embeddings)
                embeddings = embeddings.type(torch.FloatTensor)
                embeddings  = embeddings
                torch.save(embeddings, path)
        
        return peg_pca_features, embeddings

    


    @property
    def num_features(self):
        return self._num_features

    def process_structure_feature(data, args):
        pass

    def process_subgraph_feature(data, args):
        pass
