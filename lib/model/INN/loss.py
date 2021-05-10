import numpy as np
import torch.nn as nn
import torch


def nll(sample):
    return 0.5*torch.sum(torch.pow(sample, 2), dim=[1,2,3])


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample, logdet):
        # sample: (16, 128, 1, 1)
        # logdet: (16)
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        log = {"images": {},
               "scalars": {
                    "loss": loss, "reference_nll_loss": reference_nll_loss,
                    "nlogdet_loss": nlogdet_loss, "nll_loss": nll_loss,
               }}

        return loss, log