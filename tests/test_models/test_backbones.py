# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrotate.models.backbones import ReResNet


def test_reresnet_backbone():
    """Test reresnet backbone."""
    with pytest.raises(KeyError):
        # ResNet depth should be in [18, 34, 50, 101, 152]
        ReResNet(20)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ReResNet(50, num_stages=0)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ReResNet(50, num_stages=5)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations) == num_stages
        ReResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = ReResNet(50, pretrained=0)

    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        ReResNet(50, style='tensorflow')

    # test reresnet50
    model = ReResNet(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch')
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 8, 8])
    assert feat[1].shape == torch.Size([1, 512, 4, 4])
    assert feat[2].shape == torch.Size([1, 1024, 2, 2])
    assert feat[3].shape == torch.Size([1, 2048, 1, 1])

    # test reresnet18
    model = ReResNet(
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch')
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 64, 8, 8])
    assert feat[1].shape == torch.Size([1, 128, 4, 4])
    assert feat[2].shape == torch.Size([1, 256, 2, 2])
    assert feat[3].shape == torch.Size([1, 512, 1, 1])
