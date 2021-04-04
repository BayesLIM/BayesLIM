"""
Optimization module
"""
import numpy as np
import torch

from . import utils






class WeightedMSELoss:

    def __init__(self, *args, **kwargs):
        """
        """
        pass

    def forward(self, prediction, target, weight=None, params=None):
        """
        """
        diff = prediction - target
        loss = torch.sum(diff / weight) / torch.sum(weight)
        return loss






