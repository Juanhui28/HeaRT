import torch
import torch_sparse

import torch
import pytorch_indexing as pytorch_indexing

def spspmm(indexA, valueA, indexB, valueB, m, k, n, data_split=1):
    """Matrix product of two sparse tensors. Both input sparse matrices need to
    be coalesced (use the :obj:`coalesced` attribute to force).

    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of first corresponding dense matrix.
        k (int): The second dimension of first corresponding dense matrix and
            first dimension of second corresponding dense matrix.
        n (int): The second dimension of second corresponding dense matrix.
        coalesced (bool, optional): If set to :obj:`True`, will coalesce both
            input sparse matrices. (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    with torch.no_grad():
        rowA, colA = indexA
        rowB, colB = indexB
        inc = int(k//data_split) + 1
        indsA, indsB = pytorch_indexing.compare_all_elements(colA, rowB, k, data_split=data_split)
        prod_inds = torch.cat((rowA[indsA].unsqueeze(0), colB[indsB].unsqueeze(0)), dim=0)
    prod_vals = valueA[indsA]*valueB[indsB]
    return torch_sparse.coalesce(prod_inds, prod_vals, m, n)
