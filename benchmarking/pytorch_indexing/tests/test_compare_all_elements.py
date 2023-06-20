from unittest import TestCase

import pytorch_indexing
from pytorch_indexing import compare_all_elements

import torch

class Test_Compare_All_Elements(TestCase):

    def test_tensorA_equal_tensorB_data_split1(self):
        n = 1000
        tensor1 = torch.LongTensor(n).random_(0, int(n/10))
        tensor2 = torch.LongTensor(n).random_(0, int(n/10))
        inds1, inds2 = compare_all_elements(tensor1, tensor2, int(n/10))
        self.assertTrue(torch.all(torch.eq(tensor1[inds1],tensor2[inds2])))

    def test_tensorA_equal_tensorB_data_split2000(self):
        n = 1000
        tensor1 = torch.LongTensor(n).random_(0, int(n/10))
        tensor2 = torch.LongTensor(n).random_(0, int(n/10))
        inds1, inds2 = compare_all_elements(tensor1, tensor2, int(n/10), data_split=2000)
        self.assertTrue(torch.all(torch.eq(tensor1[inds1],tensor2[inds2])))

    def test_tensorA_plus_tensorB_equal_ten_data_split1(self):
        n = 1000
        tensor1 = torch.LongTensor(n).random_(0, int(n/10))
        tensor2 = torch.LongTensor(n).random_(0, int(n/10))
        tensor1_temp = tensor1 - 10
        tensor2_temp = -tensor2
        inds1, inds2 = compare_all_elements(tensor1_temp, tensor2_temp, int(n/10))
        self.assertTrue(torch.all(torch.eq(tensor1[inds1]+tensor2[inds2], torch.ones(inds1.shape[0])*10)))

    def test_tensorA_plus_tensorB_equal_ten_data_split1715(self):
        n = 1000
        tensor1 = torch.LongTensor(n).random_(0, int(n/10))
        tensor2 = torch.LongTensor(n).random_(0, int(n/10))
        tensor1_temp = tensor1 - 10
        tensor2_temp = -tensor2
        inds1, inds2 = compare_all_elements(tensor1_temp, tensor2_temp, int(n/10), data_split=1715)
        self.assertTrue(torch.all(torch.eq(tensor1[inds1]+tensor2[inds2], torch.ones(inds1.shape[0])*10)))
