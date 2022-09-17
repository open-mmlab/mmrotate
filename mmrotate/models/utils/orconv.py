# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import absolute_import
import math

import torch
import torch.nn.functional as F
from mmcv.ops import active_rotated_filter
from mmengine.utils import to_2tuple
from torch.nn.modules import Conv2d
from torch.nn.parameter import Parameter


class ORConv2d(Conv2d):
    """Oriented 2-D convolution.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): The size of kernel.
        arf_config (tuple, optional): a tuple consist of nOrientation and
            nRotation.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input.
            channels to output channels. Default: 1.
        bias (bool): If True, adds a learnable bias to the output.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 arf_config=None,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.nOrientation, self.nRotation = to_2tuple(arf_config)
        assert (math.log(self.nOrientation) + 1e-5) % math.log(2) < 1e-3, \
            f'invalid nOrientation {self.nOrientation}'
        assert (math.log(self.nRotation) + 1e-5) % math.log(2) < 1e-3, \
            f'invalid nRotation {self.nRotation}'

        super(ORConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias)
        self.register_buffer('indices', self.get_indices())
        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels, self.nOrientation,
                         *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels * self.nRotation))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of ORConv2d."""
        n = self.in_channels * self.nOrientation
        for k in self.kernel_size:
            n *= k
        self.weight.data.normal_(0, math.sqrt(2.0 / n))
        if self.bias is not None:
            self.bias.data.zero_()

    def get_indices(self):
        """Get the indices of ORConv2d."""
        kernel_indices = {
            1: {
                0: (1, ),
                45: (1, ),
                90: (1, ),
                135: (1, ),
                180: (1, ),
                225: (1, ),
                270: (1, ),
                315: (1, )
            },
            3: {
                0: (1, 2, 3, 4, 5, 6, 7, 8, 9),
                45: (2, 3, 6, 1, 5, 9, 4, 7, 8),
                90: (3, 6, 9, 2, 5, 8, 1, 4, 7),
                135: (6, 9, 8, 3, 5, 7, 2, 1, 4),
                180: (9, 8, 7, 6, 5, 4, 3, 2, 1),
                225: (8, 7, 4, 9, 5, 1, 6, 3, 2),
                270: (7, 4, 1, 8, 5, 2, 9, 6, 3),
                315: (4, 1, 2, 7, 5, 3, 8, 9, 6)
            }
        }
        delta_orientation = 360 / self.nOrientation
        delta_rotation = 360 / self.nRotation
        kH, kW = self.kernel_size
        indices = torch.IntTensor(self.nOrientation * kH * kW, self.nRotation)
        for i in range(0, self.nOrientation):
            for j in range(0, kH * kW):
                for k in range(0, self.nRotation):
                    angle = delta_rotation * k
                    layer = (i + math.floor(
                        angle / delta_orientation)) % self.nOrientation
                    kernel = kernel_indices[kW][angle][j]
                    indices[i * kH * kW + j, k] = int(layer * kH * kW + kernel)
        return indices.view(self.nOrientation, kH, kW, self.nRotation)

    def rotate_arf(self):
        """Build active rotating filter module."""
        return active_rotated_filter(self.weight, self.indices)

    def forward(self, input):
        """Forward function."""
        return F.conv2d(input, self.rotate_arf(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        arf_config = f'[{self.nOrientation}]' \
            if self.nOrientation == self.nRotation \
            else f'[{self.nOrientation}-{self.nRotation}]'
        s = ('{name}({arf_config} {in_channels}, '
             '{out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0, ) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(
            name=self.__class__.__name__,
            arf_config=arf_config,
            **self.__dict__)
