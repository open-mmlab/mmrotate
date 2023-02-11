# Copyright (c) OpenMMLab. All rights reserved.
# Modified from csuhan: https://github.com/csuhan/ReDet
from typing import Optional, Sequence, Tuple

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmrotate.registry import MODELS

try:
    import e2cnn.nn as enn
    from e2cnn.nn import EquivariantModule
    from ..utils.enn import (build_enn_divide_feature, build_enn_norm_layer,
                             build_enn_trivial_feature, ennAvgPool, ennConv,
                             ennMaxPool, ennReLU, ennTrivialConv)
except ImportError:
    enn = None
    build_enn_divide_feature = None
    build_enn_norm_layer = None
    build_enn_trivial_feature = None
    ennAvgPool = None
    ennConv = None
    ennMaxPool = None
    ennReLU = None
    ennTrivialConv = None
    EquivariantModule = BaseModule


class BasicBlock(EquivariantModule):
    """BasicBlock for ReResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1.
            Defaults to 1.
        stride (int): stride of the block. Defaults to 1
        dilation (int): dilation of convolution. Defaults to 1
        downsample (nn.Module): downsample operation on identity branch.
            Defaults to None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to
            construct and config conv layer. Defaults to None
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Defaults to ``dict(type='BN')``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 1,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        style: str = 'pytorch',
        with_cp: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN')
    ) -> None:
        super().__init__()
        self.in_type = build_enn_divide_feature(in_channels)
        self.out_type = build_enn_divide_feature(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_enn_norm_layer(
            self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_enn_norm_layer(out_channels, postfix=2)

        self.conv1 = ennConv(
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = ennReLU(self.mid_channels)
        self.conv2 = ennConv(
            self.mid_channels, out_channels, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu2 = ennReLU(out_channels)
        self.downsample = downsample

    @property
    def norm1(self) -> str:
        """Get normalizion layer's name."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> str:
        """Get normalizion layer's name."""
        return getattr(self, self.norm2_name)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of BasicBlock."""

        def _inner_forward(x: Tensor) -> Tensor:
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu2(out)

        return out

    def evaluate_output_shape(self, input_shape: Sequence) -> Sequence:
        """Evaluate output shape."""
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape


class Bottleneck(EquivariantModule):
    """Bottleneck block for ReResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2.
            Defaults to 4.
        stride (int): stride of the block. Defaults to 1
        dilation (int): dilation of convolution. Defaults to 1
        downsample (nn.Module): downsample operation on identity branch.
            Defaults to None.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to
            construct and config conv layer. Defaults to None
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Defaults to ``dict(type='BN')``
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        style: str = 'pytorch',
        with_cp: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN')
    ) -> None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.in_type = build_enn_divide_feature(in_channels)
        self.out_type = build_enn_divide_feature(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_enn_norm_layer(
            self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_enn_norm_layer(
            self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_enn_norm_layer(out_channels, postfix=3)

        self.conv1 = ennConv(
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = ennReLU(self.mid_channels)
        self.conv2 = ennConv(
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu2 = ennReLU(self.mid_channels)
        self.conv3 = ennConv(
            self.mid_channels, out_channels, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu3 = ennReLU(out_channels)

        self.downsample = downsample

    @property
    def norm1(self) -> str:
        """Get normalizion layer's name."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> str:
        """Get normalizion layer's name."""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self) -> str:
        """Get normalizion layer's name."""
        return getattr(self, self.norm3_name)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of Bottleneck."""

        def _inner_forward(x: Tensor) -> Tensor:
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu2(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu3(out)

        return out

    def evaluate_output_shape(self, input_shape: Sequence) -> Sequence:
        """Evaluate output shape."""
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape


def get_expansion(block: nn.Module, expansion: Optional[int] = None) -> int:
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (nn.Module): The block class.
        expansion (int, optional): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ReResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Defaults to None.
        stride (int): stride of the first block. Defaults to 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to
            construct and config conv layer. Defaults to None
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Defaults to ``dict(type='BN')``
    """

    def __init__(self,
                 block: nn.Module,
                 num_blocks: int,
                 in_channels: int,
                 out_channels: int,
                 expansion: Optional[int] = None,
                 stride: int = 1,
                 avg_down: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 **kwargs) -> None:
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    ennAvgPool(
                        in_channels,
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True))
            downsample.extend([
                ennConv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_enn_norm_layer(out_channels)[1]
            ])
            downsample = enn.SequentialModule(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super().__init__(*layers)


@MODELS.register_module()
class ReResNet(BaseModule):
    """ReResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`_ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_channels (int): Output channels of the stem layer.
            Defaults to 64.
        base_channels (int): Middle channels of the first stage.
            Defaults to 64.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Defaults to None.
        num_stages (int): Stages of the network. Defaults to 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Defaults to ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Defaults to False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to
            construct and config conv layer. Defaults to None
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Defaults to ``dict(type='BN')``
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Defaults to True.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 in_channels: int = 3,
                 stem_channels: int = 64,
                 base_channels: int = 64,
                 expansion: Optional[int] = None,
                 num_stages: int = 4,
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 out_indices: Sequence[int] = (3, ),
                 style: str = 'pytorch',
                 deep_stem: bool = False,
                 avg_down: bool = False,
                 frozen_stages: int = -1,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 zero_init_residual: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        try:
            import e2cnn  # noqa: F401
        except ImportError:
            raise ImportError(
                'Please install e2cnn by "pip install -e '
                'git+https://github.com/QUVA-Lab/e2cnn.git#egg=e2cnn"')
        self.in_type = build_enn_trivial_feature(in_channels)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs) -> nn.Module:
        """Build Reslayer."""
        return ResLayer(**kwargs)

    @property
    def norm1(self) -> str:
        """Get normalizion layer's name."""
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Build stem layer."""
        if not self.deep_stem:
            self.conv1 = ennTrivialConv(
                in_channels, stem_channels, kernel_size=7, stride=2, padding=3)
            self.norm1_name, norm1 = build_enn_norm_layer(
                stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = ennReLU(stem_channels)
        self.maxpool = ennMaxPool(
            stem_channels, kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self) -> None:
        """Freeze stages."""
        if self.frozen_stages >= 0:
            if not self.deep_stem:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """Forward function of ReResNet."""
        if not self.deep_stem:
            x = enn.GeometricTensor(x, self.in_type)
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Train function of  ReResNet."""
        super().train(mode=mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
