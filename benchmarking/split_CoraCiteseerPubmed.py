### exampled code to split small datasets

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, to_undirected


def save(file_name, data_name, edge):
    with open('dataset/'+data_name+'/'+ file_name+ '.txt', 'w') as f:
        for i in range(edge.size(1)):
            s, t = edge[0][i].item(), edge[1][i].item()
            f.write(str(s)+'\t'+str(t) +'\n')
            f.flush()


data_name = 'cora'
dataset = Planetoid(root="dataset", name=data_name)

data = dataset[0]

### get unique edges
edge_index = data.edge_index
edge_index = to_undirected(edge_index)
edge_index = coalesce(edge_index)
mask = edge_index[0] <= edge_index[1]
edge_index = edge_index[:, mask]

### split 
perm = torch.randperm(edge_index.size(1))
test_pos_len = int(len(perm)*0.1)
valid_pos_len = int(len(perm)*0.05)
valid_pos = edge_index[:,perm[:valid_pos_len]]
test_pos = edge_index[:,perm[valid_pos_len:valid_pos_len+test_pos_len]]
train_pos = edge_index[:, perm[test_pos_len+valid_pos_len:]]

### to generate negatives 
nodenum = data.x.size(0)
edge_dict = {}
for i in range(edge_index.size(1)):
    s, t = edge_index[0][i].item(), edge_index[1][i].item()
    if s not in edge_dict: edge_dict[s] = set()
    if t not in edge_dict: edge_dict[t] = set()
    edge_dict[s].add(t)
    edge_dict[t].add(s)

### negatives should not be the positive edges
valid_neg = []
for i in range(valid_pos.size(1)):
    src = torch.randint(0, nodenum, (1,)).item()
    dst = torch.randint(0, nodenum, (1,)).item()
    while dst in edge_dict[src] or src in edge_dict[dst]:
        src = torch.randint(0, nodenum, (1,)).item()
        dst = torch.randint(0, nodenum, (1,)).item()

    valid_neg.append([src, dst])


test_neg = []
for i in range(test_pos.size(1)):
    src = torch.randint(0, nodenum, (1,)).item()
    dst = torch.randint(0, nodenum, (1,)).item()
    while dst in edge_dict[src] or src in edge_dict[dst]:
        src = torch.randint(0, nodenum, (1,)).item()
        dst = torch.randint(0, nodenum, (1,)).item()

    test_neg.append([src, dst])

valid_neg = torch.tensor(valid_neg).t()
test_neg = torch.tensor(test_neg).t()

### save data
save('train_pos', data_name, train_pos)
save('valid_pos', data_name, valid_pos)
save('valid_neg', data_name, valid_neg)

save('test_pos', data_name, test_pos)
save('test_neg', data_name, test_neg)
