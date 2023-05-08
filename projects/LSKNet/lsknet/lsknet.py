import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import (BaseModule, constant_init, normal_init,
                            trunc_normal_init)
from torch.nn.modules.utils import _pair as to_2tuple

from mmrotate.registry import MODELS


class Mlp(BaseModule):
    """An implementation of Mlp of LSKNet.

    Refer to
    mmclassification/mmcls/models/backbones/van.py.
    Args:
        in_features (int): The feature dimension. Same as
            `MultiheadAttention`.
        hidden_features (int): The hidden dimension of Mlps.
        act_cfg (dict, optional): The activation config for Mlps.
            Default: dict(type='GELU').
        drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        init_cfg (obj:`mmengine.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 init_cfg=None):
        super(Mlp, self).__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKmodule(BaseModule):
    """LSK module(LSK) of LSKNet.

    Args:
        dim (int): Number of input channels.
        init_cfg (obj:`mmengine.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, dim, init_cfg=None):
        super(LSKmodule, self).__init__(init_cfg=init_cfg)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(
            1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class Attention(BaseModule):
    """Basic attention module in LSKblock.

    Args:
        d_model (int): Number of input channels.
        init_cfg (obj:`mmengine.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, d_model, init_cfg=None):
        super(Attention, self).__init__(init_cfg=init_cfg)

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKmodule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):
    """A block of LSK.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop (float): Dropout rate after embedding. Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.1.
        act_layer (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (obj:`mmengine.ConfigDict`): The Config for normalization.
            Default: None.
        init_cfg (obj:`mmengine.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=None,
                 init_cfg=None):
        super(Block, self).__init__(init_cfg=init_cfg)
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding of LSK.

    Args:
        patch_size (int): OverlapPatchEmbed patch size. Defaults to 7
        stride (int): OverlapPatchEmbed stride. Defaults to 4
        in_chans (int): Number of input channels. Defaults to 3.
        embed_dim (int): The hidden dimension of OverlapPatchEmbed.
        norm_cfg (obj:`mmengine.ConfigDict`): The Config for normalization.
            Default: None.
        init_cfg (obj:`mmengine.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768,
                 norm_cfg=None,
                 init_cfg=None):
        super(OverlapPatchEmbed, self).__init__(init_cfg=init_cfg)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
            tuple: Contains merged results and its spatial shape.
            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - H (list[int]): Height shape of x
            - W (list[int]): Weight shape of x
        """
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


@MODELS.register_module()
class LSKNet(BaseModule):
    """Large Selective Kernel Network.

    A PyTorch implement of : `Large Selective Kernel Network for
        Remote Sensing Object Detection.`
        PDF: https://arxiv.org/pdf/2303.09030.pdf
    Inspiration from
    https://github.com/zcablii/Large-Selective-Kernel-Network
    Args:
        in_chans (int): The num of input channels. Defaults to 3.
        embed_dims (List[int]): Embedding channels of each LSK block.
            Defaults to [64, 128, 256, 512]
        mlp_ratios (List[int]): Mlp ratios. Defaults to [8, 8, 4, 4]
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        depths (List[int]): Number of LSK block in each stage.
            Defaults to [3, 4, 6, 3]
        num_stages (int): Number of stages. Defaults to 4
        pretrained (bool): If the model weight is pretrained. Defaults to None,
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to None.
    """

    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super(LSKNet, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_cfg=norm_cfg)

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg) for j in range(depths[i])
            ])
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LSKNet, self).init_weights()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DWConv(nn.Module):
    """Depth-wise convolution
    Args:
        dim (int): In/out channel of the Depth-wise convolution.
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
