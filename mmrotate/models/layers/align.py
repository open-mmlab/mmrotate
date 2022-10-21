# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmcv.ops import DeformConv2d, DeformConv2dPack, rotated_feature_align
from mmengine.model import BaseModule, normal_init
from torch import Tensor, nn

from mmrotate.registry import MODELS


@MODELS.register_module()
class AlignConv(BaseModule):
    """AlignConv."""

    def __init__(self, feat_channels, kernel_size, strides, deform_groups=1):
        super(AlignConv, self).__init__()
        self.feat_channels = feat_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.deform_conv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        """Get the offset of AlignConv."""
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = \
            x_ctr / stride, y_ctr / stride, \
            w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        offset = offset.reshape(anchors.size(0),
                                -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward_single(self, x, anchor, stride):
        """Forward function for single level."""
        num_imgs, _, H, W = x.shape
        offset_list = [
            self.get_offset(anchor[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.deform_conv(x, offset_tensor)
        x = self.relu(x)
        return x

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function."""
        mlvl_anchors = []
        for i in range(len(x)):
            anchor = torch.stack([anchor[i] for anchor in anchors], dim=0)
            mlvl_anchors.append(anchor)
        out = []
        for x, anchor, stride in zip(x, mlvl_anchors, self.strides):
            out.append(self.forward_single(x, anchor, stride))
        return out


@MODELS.register_module()
class PseudoAlignModule(BaseModule):
    """Pseudo Align Module."""

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function."""
        return x


@MODELS.register_module()
class DCNAlignModule(DeformConv2dPack):
    """DCN Align Module.

    All args are from DeformConv2dPack.
    TODO: maybe use build_conv_layer is more flexible.
    """

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function."""
        return [super(DCNAlignModule, self).forward(xi) for xi in x]


@MODELS.register_module()
class FRM(BaseModule):
    """Feature refine module for `R3Det`.

    Args:
        feat_channels (int): Number of input channels.
        strides (list[int]): The strides of featmap.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
    """

    def __init__(self, feat_channels: int, strides: List[int]) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.strides = strides
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of feature refine module."""
        self.conv_5_1 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0))
        self.conv_1_5 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2))
        self.conv_1_1 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=1)

    def init_weights(self) -> None:
        """Initialize weights of feature refine module."""
        normal_init(self.conv_5_1, std=0.01)
        normal_init(self.conv_1_5, std=0.01)
        normal_init(self.conv_1_1, std=0.01)

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function.

        Args:
            x (list[Tensor]): feature maps of multiple scales
            anchors (list[list[Tensor]]): anchors of multiple
                scales of multiple images

        Returns:
            list[Tensor]: refined feature maps of multiple scales.
        """
        mlvl_rbboxes = [torch.cat(best_rbbox) for best_rbbox in zip(*anchors)]
        out = []
        for x_scale, best_rbboxes_scale, fr_scale in zip(
                x, mlvl_rbboxes, self.strides):
            feat_scale_1 = self.conv_5_1(self.conv_1_5(x_scale))
            feat_scale_2 = self.conv_1_1(x_scale)
            feat_scale = feat_scale_1 + feat_scale_2
            feat_refined_scale = rotated_feature_align(feat_scale,
                                                       best_rbboxes_scale,
                                                       1 / fr_scale)
            out.append(x_scale + feat_refined_scale)
        return out
