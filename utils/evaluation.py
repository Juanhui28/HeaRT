"""
hitrate@k, mean reciprocal rank (MRR) and Area under the receiver operator characteristic curve (AUC) evaluation metrics
"""
from sklearn.metrics import roc_auc_score


# @torch.no_grad()
# def test(model, evaluator, train_loader, val_loader, test_loader, args, device, emb=None, eval_metric='hits'):
#     print('starting testing')
#     t0 = time.time()
#     model.eval()
#     print("get train predictions")
#     test_func = get_test_func(args.model)
#     pos_train_pred, neg_train_pred, train_pred, train_true = test_func(model, train_loader, device, args, split='train')
#     print("get val predictions")
#     pos_val_pred, neg_val_pred, val_pred, val_true = test_func(model, val_loader, device, args, split='val')
#     print("get test predictions")
#     pos_test_pred, neg_test_pred, test_pred, test_true = test_func(model, test_loader, device, args, split='test')

#     if eval_metric == 'hits':
#         results = evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
#                                 neg_test_pred, Ks=[args.K])
#     elif eval_metric == 'mrr':

#         results = evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
#                                neg_test_pred)
#     elif eval_metric == 'auc':
#         results = evaluate_auc(val_pred, val_true, test_pred, test_true)

#     print(f'testing ran in {time.time() - t0}')

#     return results


@torch.no_grad()
def test(algorithm, evaluator, train_dataset, val_dataset, test_dataset, args, device, emb=None, eval_metric='hits'):
    print('starting testing')
    t0 = time.time()
    model.eval()
    print("get train predictions")
    pos_train_pred, neg_train_pred, train_pred, train_true = algorithm.test(train_dataset, device, args, split='train')
    print("get val predictions")
    pos_val_pred, neg_val_pred, val_pred, val_true = algorithm.test(val_dataset, device, args, split='val')
    print("get test predictions")
    pos_test_pred, neg_test_pred, test_pred, test_true = algorithm.test(test_dataset, device, args, split='test')

    if eval_metric == 'hits':
        results = evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                                neg_test_pred, Ks=[args.K])
    elif eval_metric == 'mrr':
        results = evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                               neg_test_pred)
    elif eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    print(f'testing ran in {time.time() - t0}')

    return results



def evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,
                  Ks=[20, 50, 100], use_val_negs_for_train=True):
    """
    Evaluate the hit rate at K
    :param evaluator: an ogb Evaluator object
    :param pos_val_pred: Tensor[val edges]
    :param neg_val_pred: Tensor[neg val edges]
    :param pos_test_pred: Tensor[test edges]
    :param neg_test_pred: Tensor[neg test edges]
    :param Ks: top ks to evaluatate for
    :return: dic[ks]
    """
    results = {}
    # As the training performance is used to assess overfitting it can help to use the same set of negs for
    # train and val comparisons.
    if use_val_negs_for_train:
        neg_train = neg_val_pred
    else:
        neg_train = neg_train_pred
    for K in Ks:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_train,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

def get_split_samples(split, args, dataset_len):
    """
    get the
    :param split: train, val, test
    :param args: Namespace object
    :param dataset_len: total size of dataset
    :return:
    """
    samples = dataset_len
    if split == 'train':
        if args.dynamic_train:
            samples = get_num_samples(args.train_samples, dataset_len)
    elif split in {'val', 'valid'}:
        if args.dynamic_val:
            samples = get_num_samples(args.val_samples, dataset_len)
    elif split == 'test':
        if args.dynamic_test:
            samples = get_num_samples(args.test_samples, dataset_len)
    else:
        raise NotImplementedError(f'split: {split} is not a valid split')
    return samples

def evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    """
    Evaluate the mean reciprocal rank at K
    :param evaluator: an ogb Evaluator object
    :param pos_val_pred: Tensor[val edges]
    :param neg_val_pred: Tensor[neg val edges]
    :param pos_test_pred: Tensor[test edges]
    :param neg_test_pred: Tensor[neg test edges]
    :param Ks: top ks to evaluatate for
    :return: dic with single key 'MRR'
    """
    neg_train_pred = neg_train_pred.view(pos_train_pred.shape[0], -1)
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}

    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        # for mrr negs all have the same src, so can't use the val negs, but as long as the same  number of negs / pos are
        # used the results will be comparable.
        'y_pred_neg': neg_train_pred,
    })['mrr_list'].mean().item()

    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (train_mrr, valid_mrr, test_mrr)

    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    """
    the ROC AUC
    :param val_pred: Tensor[val edges] predictions
    :param val_true: Tensor[val edges] labels
    :param test_pred: Tensor[test edges] predictions
    :param test_true: Tensor[test edges] labels
    :return:
    """
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results
