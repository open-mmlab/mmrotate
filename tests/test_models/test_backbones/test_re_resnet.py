# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import e2cnn.nn as enn
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmrotate.models.backbones.re_resnet import (BasicBlock, Bottleneck,
                                                 ReResNet, ResLayer)
from mmrotate.models.utils.enn import build_enn_divide_feature


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (BasicBlock, Bottleneck)):
        return True
    return False


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


class TestReFPN(TestCase):

    def test_re_resnet_basic_block(self):

        # test BasicBlock structure and forward
        block = BasicBlock(64, 64)
        self.assertIsInstance(block.conv1, enn.R2Conv)
        self.assertEqual(block.conv1.kernel_size, 3)
        self.assertIsInstance(block.conv2, enn.R2Conv)
        self.assertEqual(block.conv2.kernel_size, 3)
        x = torch.randn(1, 64, 56, 56)
        x = enn.GeometricTensor(x, build_enn_divide_feature(64))
        x_out = block(x)
        assert x_out.shape == torch.Size([1, 64, 56, 56])

    def test_re_resnet_bottleneck(self):
        with self.assertRaises(AssertionError):
            # Style must be in ['pytorch', 'caffe']
            Bottleneck(64, 64, style='tensorflow')

        block = Bottleneck(64, 64)
        x = torch.randn(1, 64, 56, 56)
        x = enn.GeometricTensor(x, build_enn_divide_feature(64))
        x_out = block(x)
        assert x_out.shape == torch.Size([1, 64, 56, 56])

        # Test Bottleneck style
        block = Bottleneck(64, 64, stride=2, style='pytorch')
        self.assertEqual(block.conv1.stride, 1)
        self.assertEqual(block.conv2.stride, 2)
        block = Bottleneck(64, 64, stride=2, style='caffe')
        self.assertEqual(block.conv1.stride, 2)
        self.assertEqual(block.conv2.stride, 1)

    def test_re_resnet_res_layer(self):
        # Test ResLayer of 3 Bottleneck w\o downsample
        layer = ResLayer(Bottleneck, 3, 64, 64)
        assert len(layer) == 3
        for i in range(len(layer)):
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 64, 56, 56)
        x = enn.GeometricTensor(x, build_enn_divide_feature(64))
        x_out = layer(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

    def test_re_resnet_backbone(self):
        """Test reresnet backbone."""
        with self.assertRaises(KeyError):
            # ResNet depth should be in [18, 34, 50, 101, 152]
            ReResNet(20)

        with self.assertRaises(AssertionError):
            # In ResNet: 1 <= num_stages <= 4
            ReResNet(50, num_stages=0)

        with self.assertRaises(AssertionError):
            # In ResNet: 1 <= num_stages <= 4
            ReResNet(50, num_stages=5)

        with self.assertRaises(AssertionError):
            # len(strides) == len(dilations) == num_stages
            ReResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

        with self.assertRaises(TypeError):
            # pretrained must be a string path
            model = ReResNet(50, pretrained=0)

        with self.assertRaises(AssertionError):
            # Style must be in ['pytorch', 'caffe']
            ReResNet(50, style='tensorflow')

        # test re_resnet50
        model = ReResNet(
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            style='pytorch')
        model.train()

        imgs = torch.randn(1, 3, 32, 32)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, torch.Size([1, 256, 8, 8]))
        self.assertEqual(feat[1].shape, torch.Size([1, 512, 4, 4]))
        self.assertEqual(feat[2].shape, torch.Size([1, 1024, 2, 2]))
        self.assertEqual(feat[3].shape, torch.Size([1, 2048, 1, 1]))

        # test re_resnet18
        model = ReResNet(
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            style='pytorch')
        model.train()

        imgs = torch.randn(1, 3, 32, 32)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, torch.Size([1, 64, 8, 8]))
        self.assertEqual(feat[1].shape, torch.Size([1, 128, 4, 4]))
        self.assertEqual(feat[2].shape, torch.Size([1, 256, 2, 2]))
        self.assertEqual(feat[3].shape, torch.Size([1, 512, 1, 1]))
