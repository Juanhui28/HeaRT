import torch
import warnings
warnings.filterwarnings("ignore") #until PyTorch patches nonzero

def compare_all_elements(tensorA, tensorB, max_val, data_split=1):
    """
    Description.....
    
    Parameters:
        tensorA:         first array to be compared (1D torch.tensor of ints)
        tensorB:         second array to be compared (1D torch.tensor of ints)
        max_val:         the largest element in either tensorA or tensorB (real number)
        data_split:      the number of subsets to split the mask up into (int)
    Returns:
        compared_indsA:  indices of tensorA that match elements in tensorB (1D torch.tensor of ints, type torch.long)
        compared_indsB:  indices of tensorB that match elements in tensorA (1D torch.tensor of ints, type torch.long)
    """
    compared_indsA, compared_indsB, inc = torch.tensor([]).to(tensorA.device), torch.tensor([]).to(tensorA.device), int(max_val//data_split) + 1
    for iii in range(data_split):
        indsA, indsB = (iii*inc<=tensorA)*(tensorA<(iii+1)*inc), (iii*inc<=tensorB)*(tensorB<(iii+1)*inc)
        tileA, tileB = tensorA[indsA], tensorB[indsB]
        tileA, tileB = tileA.unsqueeze(0).repeat(tileB.size(0), 1), torch.transpose(tileB.unsqueeze(0), 0, 1).repeat(1, tileA.size(0))
        nz_inds = torch.nonzero(tileA == tileB, as_tuple=False)
        nz_indsA, nz_indsB = nz_inds[:, 1], nz_inds[:, 0]
        compared_indsA, compared_indsB = torch.cat((compared_indsA, indsA.nonzero()[nz_indsA]), 0), torch.cat((compared_indsB, indsB.nonzero()[nz_indsB]), 0)
    return compared_indsA.squeeze().long(), compared_indsB.squeeze().long()
