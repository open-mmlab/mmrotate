# Copyright (c) OpenMMLab. All rights reserved.
# Modified from csuhan: https://github.com/csuhan/ReDet
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from mmdet.utils import MultiConfig, OptConfigType
from mmengine.model import BaseModule
from torch import Tensor

from mmrotate.registry import MODELS

try:
    import e2cnn.nn as enn  # noqa: F401
    from e2cnn.nn import EquivariantModule
    from ..utils.enn import (build_enn_feature, build_enn_norm_layer, ennConv,
                             ennInterpolate, ennMaxPool, ennReLU)
except ImportError:
    build_enn_feature = None
    build_enn_norm_layer = None
    ennConv = None
    ennInterpolate = None
    ennMaxPool = None
    ennReLU = None
    EquivariantModule = BaseModule


class ConvModule(EquivariantModule):
    """ConvModule.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int): The size of kernel.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Defaults to 0.
        dilation (int or tuple): Spacing between kernel elements.
            Defaults to 1.
        groups (int): Number of blocked connections from input.
            channels to output channels. Defaults to 1.
        bias (bool): If True, adds a learnable bias to the output.
            Defaults to `auto`.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to
            construct and config conv layer. Defaults to None
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Defaults to None
        activation (str): Activation layer in ConvModule.
            Defaults to 'relu'.
        inplace (bool): can optionally do the operation in-place.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: str = 'auto',
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        activation: str = 'relu',
        inplace: bool = False,
        order: Tuple[str] = ('conv', 'norm', 'act')
    ) -> None:
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.in_type = build_enn_feature(in_channels)
        self.out_type = build_enn_feature(out_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        # build convolution layer
        self.conv = ennConv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = padding
        self.groups = groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            if conv_cfg is not None and conv_cfg['type'] == 'ORConv':
                norm_channels = int(norm_channels * 8)
            self.norm_name, norm = build_enn_norm_layer(norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError(
                    f'{self.activation} is currently not supported.')
            if self.activation == 'relu':
                self.activate = ennReLU(out_channels)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self) -> str:
        """Get normalizion layer's name."""
        return getattr(self, self.norm_name)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        nonlinearity = 'relu' if self.activation is None \
            else self.activation  # noqa: F841

    def forward(self,
                x: Tensor,
                activate: bool = True,
                norm: bool = True) -> Tensor:
        """Forward function of ConvModule."""
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x

    def evaluate_output_shape(self, input_shape: Sequence) -> Sequence:
        """Evaluate output shape."""
        return input_shape


@MODELS.register_module()
class ReFPN(BaseModule):
    """ReFPN.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level
            used to build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level
            (exclusive) to build the feature pyramid. Defaults to -1,
            which means the last level.
        add_extra_convs (bool): It decides whether to add conv layers
            on top of the original feature maps. Default to False.
        extra_convs_on_inputs (bool): It specifies the source feature
            map of the extra convs is the last feat map of neck inputs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to
            construct and config conv layer. Defaults to None
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Defaults to None
        activation (str, optional): Activation layer in ConvModule.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: bool = False,
        extra_convs_on_inputs: bool = True,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        activation: Optional[str] = None,
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        try:
            import e2cnn  # noqa: F401
        except ImportError:
            raise ImportError(
                'Please install e2cnn by "pip install -e '
                'git+https://github.com/QUVA-Lab/e2cnn.git#egg=e2cnn"')
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            up_sample = ennInterpolate(out_channels, 2)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.up_samples.append(up_sample)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.max_pools = nn.ModuleList()
        self.relus = nn.ModuleList()

        used_backbone_levels = len(self.lateral_convs)
        if self.num_outs > used_backbone_levels:
            # use max pool to get more levels on top of outputs
            # (e.g., Rotated Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    self.max_pools.append(
                        ennMaxPool(out_channels, 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                for i in range(used_backbone_levels + 1, self.num_outs):
                    self.relus.append(ennReLU(out_channels))

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function of ReFPN."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += self.up_samples[i](laterals[i])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Rotated Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(self.max_pools[i](outs[-1]))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](self.relus[i](outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # convert to tensor
        outs = [out.tensor for out in outs]

        return tuple(outs)
