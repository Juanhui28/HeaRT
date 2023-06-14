from unittest import TestCase

import torch
import pytorch_indexing
from pytorch_indexing import spspmm
from torch_sparse import coalesce

class Test_SPSPMM(TestCase):

    def test_spspmm_autograd_setvals(self):
        indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
        valueA = torch.tensor([1, 2, 3, 4, 5])
        indexB = torch.tensor([[0, 2], [1, 0]])
        valueB = torch.tensor([2, 4])

        indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2,data_split=1)
        self.assertTrue(indexC.tolist() == [[0, 1, 2], [0, 1, 1]] and valueC.tolist() == [8, 6, 8])

    def test_spspmm_autograd_setvals_data_split21(self):
        indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
        valueA = torch.tensor([1, 2, 3, 4, 5])
        indexB = torch.tensor([[0, 2], [1, 0]])
        valueB = torch.tensor([2, 4])

        indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2, data_split=21)
        self.assertTrue(indexC.tolist() == [[0, 1, 2], [0, 1, 1]] and valueC.tolist() == [8, 6, 8])

    def test_spspmm_matches_cuda_vals_datasplit1(self):
        n = 7
        nz = 2**n
        vals1 = torch.rand(nz, requires_grad=True)
        inds1 = torch.LongTensor(2,nz).random_(0, 2**n)
        inds1, vals1 = coalesce(inds1, vals1, 2**n, 2**n)
        vals2 = torch.rand(nz, requires_grad=True)
        inds2 = torch.LongTensor(2,nz).random_(0, 2**n)
        inds2, vals2 = coalesce(inds2, vals2, 2**n, 2**n)
        my_prod_inds, my_prod_vals = spspmm(inds1, vals1, inds2, vals2, 2**n, 2**n, 2**n)
        prod_inds, prod_vals = spspmm(inds1, vals1, inds2, vals2, 2**n, 2**n, 2**n)
        self.assertTrue(torch.allclose(prod_vals, my_prod_vals) and torch.all(torch.eq(prod_inds, my_prod_inds)))

    def test_spspmm_matches_cuda_vals_datasplit17(self):
        n = 7
        nz = 2**n
        vals1 = torch.rand(nz, requires_grad=True)
        inds1 = torch.LongTensor(2,nz).random_(0, 2**n)
        inds1, vals1 = coalesce(inds1, vals1, 2**n, 2**n)
        vals2 = torch.rand(nz, requires_grad=True)
        inds2 = torch.LongTensor(2,nz).random_(0, 2**n)
        inds2, vals2 = coalesce(inds2, vals2, 2**n, 2**n)
        my_prod_inds, my_prod_vals = spspmm(inds1, vals1, inds2, vals2, 2**n, 2**n, 2**n, data_split=17)
        prod_inds, prod_vals = spspmm(inds1, vals1, inds2, vals2, 2**n, 2**n, 2**n)
        self.assertTrue(torch.allclose(prod_vals, my_prod_vals) and torch.all(torch.eq(prod_inds, my_prod_inds)))
