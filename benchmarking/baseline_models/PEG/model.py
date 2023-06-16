import torch.nn.functional as F
import torch
# from dataset import *
from baseline_models.PEG.layer import PEGConv
from torch import nn

class Net(torch.nn.Module):
    def __init__(self, in_feats_dim, pos_dim, hidden_dim, no_pe, use_former_information = False):
        super(Net, self).__init__()
        
        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.use_former_information = use_former_information
        self.no_pe = no_pe
        
        self.conv1 = PEGConv(in_feats_dim = in_feats_dim, pos_dim = pos_dim, out_feats_dim = hidden_dim, no_pe=no_pe)
        self.conv2 = PEGConv(in_feats_dim = in_feats_dim, pos_dim = pos_dim, out_feats_dim = hidden_dim, no_pe=no_pe)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        if no_pe: 
            self.fc = nn.Linear(1, 1)
        else:
            self.fc = nn.Linear(2, 1)

       

    def forward(self, x, edge_index, idx):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)


        ############
        pos_dim = self.pos_dim
        
        nodes_first = x[ : , pos_dim: ][idx[0]]
        nodes_second = x[ : , pos_dim: ][idx[1]]
        pos_first = x[ : , :pos_dim ][idx[0]]
        pos_second = x[ : , :pos_dim ][idx[1]]
        
        positional_encoding = ((pos_first - pos_second)**2).sum(dim=-1, keepdim=True)

        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        out = self.fc(torch.cat([pred.reshape(len(pred), 1),positional_encoding.reshape(len(positional_encoding), 1)], 1))

        return out

    
    def get_emb(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


    def score(self, x, idx):

        
        pos_dim = self.pos_dim
        
        nodes_first = x[ : , pos_dim: ][idx[0]]
        nodes_second = x[ : , pos_dim: ][idx[1]]
        pos_first = x[ : , :pos_dim ][idx[0]]
        pos_second = x[ : , :pos_dim ][idx[1]]
        
        positional_encoding = ((pos_first - pos_second)**2).sum(dim=-1, keepdim=True)

        pred = torch.sum(nodes_first * nodes_second, dim=-1)

        if self.no_pe:
            # print('no pe 222222')
            
            out = self.fc(pred.reshape(len(pred), 1))

        else:
            out = self.fc(torch.cat([pred.reshape(len(pred), 1),positional_encoding.reshape(len(positional_encoding), 1)], 1))

        return out

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

