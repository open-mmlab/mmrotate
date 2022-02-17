# Copyright (c) OpenMMLab. All rights reserved.
import e2cnn.nn as enn
from e2cnn import gspaces

N = 8
gspace = gspaces.Rot2dOnR2(N=N)


def build_enn_divide_feature(planes):
    """build a enn regular feature map with the specified number of channels
    divided by N."""
    assert gspace.fibergroup.order() > 0
    N = gspace.fibergroup.order()
    planes = planes / N
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def build_enn_feature(planes):
    """build a enn regular feature map with the specified number of
    channels."""
    return enn.FieldType(gspace, planes * [gspace.regular_repr])


def build_enn_trivial_feature(planes):
    """build a enn trivial feature map with the specified number of
    channels."""
    return enn.FieldType(gspace, planes * [gspace.trivial_repr])


def build_enn_norm_layer(num_features, postfix=''):
    """build an enn normalizion layer."""
    in_type = build_enn_divide_feature(num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)


def ennConv(inplanes,
            outplanes,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
            dilation=1):
    """enn convolution."""
    in_type = build_enn_divide_feature(inplanes)
    out_type = build_enn_divide_feature(outplanes)
    return enn.R2Conv(
        in_type,
        out_type,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


def ennTrivialConv(inplanes,
                   outplanes,
                   kernel_size=3,
                   stride=1,
                   padding=0,
                   groups=1,
                   bias=False,
                   dilation=1):
    """enn convolution with trivial input featurn."""
    in_type = build_enn_trivial_feature(inplanes)
    out_type = build_enn_divide_feature(outplanes)
    return enn.R2Conv(
        in_type,
        out_type,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


def ennReLU(inplanes):
    """enn ReLU."""
    in_type = build_enn_divide_feature(inplanes)
    return enn.ReLU(in_type, inplace=False)


def ennAvgPool(inplanes,
               kernel_size=1,
               stride=None,
               padding=0,
               ceil_mode=False):
    """enn Average Pooling."""
    in_type = build_enn_divide_feature(inplanes)
    return enn.PointwiseAvgPool(
        in_type,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode)


def ennMaxPool(inplanes, kernel_size, stride=1, padding=0):
    """enn Max Pooling."""
    in_type = build_enn_divide_feature(inplanes)
    return enn.PointwiseMaxPool(
        in_type, kernel_size=kernel_size, stride=stride, padding=padding)


def ennInterpolate(inplanes,
                   scale_factor,
                   mode='nearest',
                   align_corners=False):
    """enn Interpolate."""
    in_type = build_enn_divide_feature(inplanes)
    return enn.R2Upsampling(
        in_type, scale_factor, mode=mode, align_corners=align_corners)
