import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import numpy as np
import torch_sparse
import torch.nn.functional as F

class GCN(MessagePassing):
    def __init__(self, args, num_hiddens):
        super(GCN, self).__init__()
        self.args = args
        self.lins = nn.ModuleList([])
        for i in range(len(num_hiddens) - 1):
            self.lins.append(nn.Linear(num_hiddens[i], num_hiddens[i+1])) 
        self.intermediate_record = []

    def forward(self, dataset, edge_index, edge_weight, is_retain_hidden=False):
        # links: [2, num_edges]
        self.intermediate_record = []
        for i, lin in enumerate(self.lins):
            if is_retain_hidden: self.intermediate_record.append(hidden)
            hidden = torch_sparse.spmm(edge_index, edge_weight, self.args.num_node, self.args.num_node, hidden) 
            # hidden = self.propagate(edge_index, x=hidden, edge_weight=edge_weight)
            hidden = lin(hidden)
            if i != (len(self.lins) - 1):
                hidden = F.relu(hidden)
            if is_retain_hidden: self.intermediate_record.append(hidden)
        # TODO: decoder part


        hidden_link = torch.matmul(links, xxx)
        
        

        return logits

    
