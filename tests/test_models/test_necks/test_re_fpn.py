# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import e2cnn.nn as enn
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmrotate.models.necks import ReFPN
from mmrotate.models.utils.enn import build_enn_divide_feature


class TestReFPN(TestCase):

    def test_refpn(self):
        """Tests fpn."""
        s = 64
        in_channels = [8, 16, 32, 64]
        feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
        out_channels = 8

        # end_level=-1 is equal to end_level=3
        ReFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=0,
            end_level=-1,
            num_outs=5)
        ReFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=0,
            end_level=3,
            num_outs=3)

        # `num_outs` is not equal to end_level - start_level + 1
        with self.assertRaises(AssertionError):
            ReFPN(
                in_channels=in_channels,
                out_channels=out_channels,
                start_level=1,
                end_level=2,
                num_outs=3)

        # `num_outs` is not equal to len(in_channels) - start_level
        with self.assertRaises(AssertionError):
            ReFPN(
                in_channels=in_channels,
                out_channels=out_channels,
                start_level=1,
                num_outs=2)

        # `end_level` is larger than len(in_channels) - 1
        with self.assertRaises(AssertionError):
            ReFPN(
                in_channels=in_channels,
                out_channels=out_channels,
                start_level=1,
                end_level=4,
                num_outs=2)

        # `num_outs` is not equal to end_level - start_level
        with self.assertRaises(AssertionError):
            ReFPN(
                in_channels=in_channels,
                out_channels=out_channels,
                start_level=1,
                end_level=3,
                num_outs=1)

        fpn_model = ReFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            add_extra_convs=True,
            num_outs=5)

        # ReFPN expects a multiple levels of features per image
        feats = [
            enn.GeometricTensor(
                torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i]),
                build_enn_divide_feature(in_channels[i]))
            for i in range(len(in_channels))
        ]
        outs = fpn_model(feats)
        self.assertEqual(len(outs), fpn_model.num_outs)
        for i in range(fpn_model.num_outs):
            outs[i].shape[1] == out_channels
            outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

        # Tests for fpn with no extra convs (pooling is used instead)
        fpn_model = ReFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            add_extra_convs=False,
            num_outs=5)
        outs = fpn_model(feats)
        self.assertEqual(len(outs), fpn_model.num_outs)
        self.assertFalse(fpn_model.add_extra_convs)
        for i in range(fpn_model.num_outs):
            outs[i].shape[1] == out_channels
            outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

        # Tests for fpn with lateral bns
        fpn_model = ReFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            add_extra_convs=True,
            no_norm_on_lateral=False,
            norm_cfg=dict(type='BN', requires_grad=True),
            num_outs=5)
        outs = fpn_model(feats)
        self.assertEqual(len(outs), fpn_model.num_outs)
        for i in range(fpn_model.num_outs):
            outs[i].shape[1] == out_channels
            outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
        bn_exist = False
        for m in fpn_model.modules():
            if isinstance(m, _BatchNorm):
                bn_exist = True
        self.assertTrue(bn_exist)
