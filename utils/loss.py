import torch

def auc_loss(logits, y, num_neg=1):
    pos_out = logits[y == 1]
    neg_out = logits[y == 0]
    # hack, should really pair negative and positives in the training set
    if len(neg_out) <= len(pos_out):
        pos_out = pos_out[:len(neg_out)]
    else:
        neg_out = neg_out[:len(pos_out)]
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).sum()


def bce_loss(logits, y, num_neg=1):
    return BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))


def get_loss(loss_str):
    if loss_str == 'bce':
        loss = bce_loss
    elif loss_str == 'auc':
        loss = auc_loss
    else:
        raise NotImplementedError
    return loss