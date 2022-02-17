# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn


class RotationInvariantPooling(nn.Module):
    """Rotating invariant pooling module."""

    def __init__(self, nInputPlane, nOrientation=8):
        super(RotationInvariantPooling, self).__init__()
        self.nInputPlane = nInputPlane
        self.nOrientation = nOrientation

    def forward(self, x):
        """Forward function."""
        N, c, h, w = x.size()
        x = x.view(N, -1, self.nOrientation, h, w)
        x, _ = x.max(dim=2, keepdim=False)
        return x
