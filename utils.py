import torch

def positional_encoding(vec, L=5):
    """ Postionally encode a batch of numbers
        
        Inputs -
            vec [N]: Batch of features to perform positional encoding on
            L: Number of terms to use in positional encoding
        Outputs -
            res [N * L]: Encoded features
    """
    powers = torch.pow(2, torch.arange(L))
    x = torch.pi*torch.unsqueeze(vec, dim=1)*powers

    return torch.concat((torch.sin(x), torch.cos(x)), dim=1)
