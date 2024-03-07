import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def evaluate_hits(evaluator, pos_pred, neg_pred, k_list):
    results = {}
    for K in k_list:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']
        hits = round(hits, 4)

        results[f'Hits@{K}'] = hits

    return results
        


def evaluate_mrr(evaluator, pos_pred, neg_pred, k_list):

    '''
        compute mrr
        neg_pred is an array with shape (batch size, num_entities_neg).
        pos_pred is an array with shape (batch size, )
    '''
    
    neg_pred = neg_pred.view(pos_pred.shape[0], -1)
    pos_pred = pos_pred.view(-1, 1)
    # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    
    results =  eval_mrr(pos_pred, neg_pred, k_list)

    return results





def eval_mrr(pos_pred, neg_pred, k_list):
    """
    Eval on hard negatives
    """

    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (neg_pred >= pos_pred).sum(dim=-1)

    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (neg_pred > pos_pred).sum(dim=-1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    results = {}
    for k in k_list:
        mean_score = (ranking_list <= k).to(torch.float).mean().item()
        results[f'Hits@{k}'] = round(mean_score, 4)

    mean_mrr = 1./ranking_list.to(torch.float)
    results['MRR'] = round(mean_mrr.mean().item(), 4)

    return results

