import torch
import torch.nn as nn


def NMSE_cuda(x_hat, x):
    power = torch.sum(x**2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == "mean":
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


def NMSE_Split_cuda(x_hat, x):
    power = torch.sum(x_hat**2, dim=(0, 2))
    mse = torch.sum((x - x_hat) ** 2, dim=(0, 2))
    nmse = mse / power
    return nmse


class NMSELossSplit(nn.Module):
    def __init__(self, reduction="mean"):
        super(NMSELossSplit, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        return NMSE_Split_cuda(x_hat, x)
