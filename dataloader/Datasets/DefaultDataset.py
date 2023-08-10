
# SAGE
# original setting
class DefaultDataset(LPDataset):
    def __init__(self, args): 
        # TODO: add neccerary links
        super(LPDataset, self).__init__(root, split, data, pos_edges, neg_edges, use_feature, use_coalesce, directed)
        self.links = torch.cat([self.pos_edges, self.neg_edges], 0)  # [n_edges, 2]
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        return (src, dst), y
    # make sure the output are two term, the first is the inputs, the second should be label