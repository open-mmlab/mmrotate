# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrotate.models.necks.re_fpn import ReFPN, enn
from mmrotate.models.utils import build_enn_divide_feature


def test_fpn():
    """Tests fpn."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]
    out_channels = 8
    # `num_outs` is not equal to len(in_channels) - start_level
    with pytest.raises(AssertionError):
        ReFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            num_outs=2)

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        ReFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=4,
            num_outs=2)

    # `num_outs` is not equal to end_level - start_level
    with pytest.raises(AssertionError):
        ReFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=3,
            num_outs=1)

    fpn_model = ReFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs=False,
        start_level=1,
        num_outs=5)

    # ReFPN expects a multiple levels of features per image
    feats = [
        enn.GeometricTensor(
            torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i]),
            build_enn_divide_feature(inplanes))
        for i, inplanes in enumerate(in_channels)
    ]

    outs = fpn_model(feats)
    assert fpn_model.add_extra_convs is False
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'inputs'
    fpn_model = ReFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_input',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_input'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'laterals'
    fpn_model = ReFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_lateral',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_lateral'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'outputs'
    fpn_model = ReFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_output',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_output'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # num_outs > len(out) & activation
    fpn_model = ReFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        activation='relu',
        start_level=1,
        num_outs=6)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
