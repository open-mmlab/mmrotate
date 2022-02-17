# Copyright (c) OpenMMLab. All rights reserved.
"""
CommandLine:
    pytest tests/test_utils/test_rutils.py
    xdoctest tests/test_utils/test_rutils.py zero
"""
import torch


def test_rotated_anchor_inside_flags():
    """Test rotated anchor inside flags."""
    from mmrotate.core.anchor import rotated_anchor_inside_flags
    flat_ranchors = torch.tensor([[0., 0., 10., 10., 0.],
                                  [95., 0., 10., 10., 0.],
                                  [0., 100., 10., 10., 0.],
                                  [101., 100., 10., 10., 0.]])
    valid_flags = torch.tensor([1, 1, 0, 1])
    img_shape = (100, 100, 3)
    inside_flags = rotated_anchor_inside_flags(flat_ranchors, valid_flags,
                                               img_shape)
    expected_inside_flags = torch.tensor([1, 1, 0, 0])

    assert torch.equal(inside_flags, expected_inside_flags)
