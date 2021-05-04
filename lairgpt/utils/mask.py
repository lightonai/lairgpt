import torch


def get_autoregressive_mask(size: int) -> torch.BoolTensor:
    return (torch.triu(torch.ones(size, size)) == 0).transpose(0, 1).contiguous()
