from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn.conv import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
# from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch import nn
from utils import glorot, zeros
import pdb
from pytorch_indexing import spspmm



class NeoGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(NeoGNN, self).__init__()

        self.args = args
        self.convs = torch.nn.ModuleList()
        # if args.data_name == 'ogbl-citation2' or  args.data_name == 'ogbl-cora' or  args.data_name == 'ogbl-citeseer':
        #     cached = False
        # elif args.data_name == 'ogbl-collab' or args.data_name == 'ogbl-ppa' or args.data_name == 'ogbl-ddi' :
        #     cached = True
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0, 0]))

        if args.data_name not in ['ogbl-ppa', 'ogbl-citation2']:
            self.f_edge = torch.nn.Sequential(torch.nn.Linear(1, args.f_edge_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.f_edge_dim, 1).double())

            self.f_node = torch.nn.Sequential(torch.nn.Linear(1, args.f_node_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.f_node_dim, 1).double())

            self.g_phi = torch.nn.Sequential(torch.nn.Linear(1, args.g_phi_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.g_phi_dim, 1).double())
        else:
            self.f_edge = torch.nn.Sequential(torch.nn.Linear(1, args.f_edge_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.f_edge_dim, 1))

            self.f_node = torch.nn.Sequential(torch.nn.Linear(1, args.f_node_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.f_node_dim, 1))

            self.g_phi = torch.nn.Sequential(torch.nn.Linear(1, args.g_phi_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.g_phi_dim, 1))
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.constant_(self.alpha, 0)
        self.f_edge.apply(self.weight_reset)
        self.f_node.apply(self.weight_reset)
        self.g_phi.apply(self.weight_reset)

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    def forward(self, edge, adj, A, x, num_nodes, predictor=None, emb=None, only_feature=False, only_structure=False, node_struct_feat=None):
                     
        batch_size = edge.shape[-1]
        # 1. compute similarity scores of node pairs via conventionl GNNs (feature + adjacency matrix)
        adj_t = adj
        out_feat = None
        if not only_structure:
            if emb is None:
                x = x
            else:
                x = emb
            for conv in self.convs[:-1]:
                x = conv(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj_t)
            if predictor is not None:
                out_feat = predictor(x[edge[0]], x[edge[1]])
            else:
                out_feat = torch.sum(x[edge[0]] * x[edge[1]], dim=0)
        
        
        
        if only_feature:
            if self.invest == 1:
                print('only use input features:' , only_feature)
            self.invest = 0

            return None, None, out_feat, None
        # 2. compute similarity scores of node pairs via Neo-GNNs
        # 2-1. Structural feature generation
        if node_struct_feat is None:
            row_A, col_A = A.nonzero()
            tmp_A = torch.stack([torch.from_numpy(row_A), torch.from_numpy(col_A)]).type(torch.LongTensor).to(edge.device)
            row_A, col_A = tmp_A[0], tmp_A[1]
            edge_weight_A = torch.from_numpy(A.data).to(edge.device)
            edge_weight_A = self.f_edge(edge_weight_A.unsqueeze(-1))
            node_struct_feat = scatter_add(edge_weight_A, col_A, dim=0, dim_size=num_nodes)
        # print('edge: ', edge.size())
        indexes_src = edge[0].cpu().numpy()
        row_src, col_src = A[indexes_src].nonzero()
        edge_index_src = torch.stack([torch.from_numpy(row_src), torch.from_numpy(col_src)]).type(torch.LongTensor).to(edge.device)
        edge_weight_src = torch.from_numpy(A[indexes_src].data).to(edge.device)
        edge_weight_src = edge_weight_src * self.f_node(node_struct_feat[col_src]).squeeze()

        indexes_dst = edge[1].cpu().numpy()
        row_dst, col_dst = A[indexes_dst].nonzero()
        edge_index_dst = torch.stack([torch.from_numpy(row_dst), torch.from_numpy(col_dst)]).type(torch.LongTensor).to(edge.device)
        edge_weight_dst = torch.from_numpy(A[indexes_dst].data).to(edge.device)
        edge_weight_dst = edge_weight_dst * self.f_node(node_struct_feat[col_dst]).squeeze()
        
        
        if self.args.data_name in ['ogbl-ppa', 'ogbl-citation2']:
            edge_index_dst = torch.stack([edge_index_dst[1], edge_index_dst[0]])
            edge_indexes, scores = spspmm(edge_index_src, edge_weight_src, edge_index_dst, edge_weight_dst, batch_size, num_nodes, batch_size, data_split=256)
            out_struct = torch.zeros(batch_size).to(edge.device)
            out_struct[edge_indexes[0][edge_indexes[0]==edge_indexes[1]]] = scores[edge_indexes[0]==edge_indexes[1]]
        else:
            mat_src = SparseTensor.from_edge_index(edge_index_src, edge_weight_src, [batch_size, num_nodes])
            mat_dst = SparseTensor.from_edge_index(edge_index_dst, edge_weight_dst, [batch_size, num_nodes])
            out_struct = (mat_src @ mat_dst.to_dense().t()).diag()
        
        out_struct = self.g_phi(out_struct.unsqueeze(-1))
        out_struct_raw = out_struct
        out_struct = torch.sigmoid(out_struct)

        if not only_structure:
            alpha = torch.softmax(self.alpha, dim=0)
            out = alpha[0] * out_struct + alpha[1] * out_feat + 1e-15
        else:
            out = None

        del edge_weight_src, edge_weight_dst, node_struct_feat
        torch.cuda.empty_cache()

        return out, out_struct, out_feat, out_struct_raw

    def forward_feature(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x


