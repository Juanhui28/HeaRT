
import torch
from utils.evaluation import get_split_samples
from utils.loss import get_loss

class GCN_algo(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = model

    @staticmethod
    def add_args(parser):
        # add some argumentation
        return parser


    # the algorithm do not need to conduct subgraphs. 
    def train_default(args, data, optimizer, device):
        # train on the original setting
        print('starting training')
        t0 = time.time()
        self.model.train()
        total_loss = 0
        # hydrate edges
        links = data.links
        labels = torch.tensor(data.labels)
        # sampling
        train_samples = get_num_samples(args.train_samples, len(labels))
        sample_indices = torch.randperm(len(labels))[:train_samples]
        links = links[sample_indices]
        labels = labels[sample_indices]

        if args.wandb:
            wandb.log({"train_total_batches": len(train_loader)})
        batch_processing_times = []
        # TODO: add the full batch selection, the current may not be correct
        args.batch_size = len(links) if args.batch_size == -1 else args.batch_size
        
        loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
        for batch_count, indices in enumerate(tqdm(loader)):
            # do node level things
            curr_links = links[indices]
            start_time = time.time()
            optimizer.zero_grad()
            logits = self.model(curr_links)

            logits = self.model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
            loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * args.batch_size
            batch_processing_times.append(time.time() - start_time)
    
    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')

    return total_loss / len(data)


    def test(self, dataset, device, args, split):
        n_samples = get_split_samples(split, args, len(dataset))
        t0 = time.time()
        preds = []
        data = dataset
        # hydrate edges
        links = data.links
        labels = torch.tensor(data.labels)
        loader = DataLoader(range(len(links)), args.eval_batch_size,
                            shuffle=False)  # eval batch size should be the largest that fits on GPU
        # get node features
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
        for batch_count, indices in enumerate(tqdm(loader)):
            curr_links = links[indices].to(device)
            batch_emb = None if emb is None else emb[curr_links].to(device)
            if args.use_struct_feature:
                subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
            else:
                subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
            batch_node_features = None if node_features is None else node_features[curr_links]
            logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
            preds.append(logits.view(-1).cpu())
            if (batch_count + 1) * args.eval_batch_size > n_samples:
                break

        if args.wandb:
            wandb.log({f"inference_{split}_epoch_time": time.time() - t0})
        pred = torch.cat(preds)
        labels = labels[:len(pred)]
        pos_pred = pred[labels == 1]
        neg_pred = pred[labels == 0]
        return pos_pred, neg_pred, pred, labels



    
