# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import normal_init
from mmcv.ops import DeformConv2d
from mmcv.runner import BaseModule
from mmcv.utils import Registry, build_from_cfg
from torch import nn

ALIGN_MODULE = Registry('Match Cost')


def build_align_module(cfg, default_args=None):
    """Builder of AlignModule."""
    return build_from_cfg(cfg, ALIGN_MODULE, default_args)


@ALIGN_MODULE.register_module()
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

    def forward(self, x, anchors):
        """Forward function."""
        mlvl_anchors = []
        for i in range(len(x)):
            anchor = torch.stack([anchor[i] for anchor in anchors], dim=0)
            mlvl_anchors.append(anchor)
        out = []
        for x, anchor, stride in zip(x, mlvl_anchors, self.strides):
            out.append(self.forward_single(x, anchor, stride))
        return out
