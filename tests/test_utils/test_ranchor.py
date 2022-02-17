# Copyright (c) OpenMMLab. All rights reserved.
"""
CommandLine:
    pytest tests/test_utils/test_ranchor.py
    xdoctest tests/test_utils/test_ranchor.py zero
"""
import torch


def test_standard_prior_generator():
    """Test standard prior generator."""
    from mmrotate.core.anchor import build_prior_generator
    anchor_generator_cfg = dict(
        type='RotatedAnchorGenerator',
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8])

    anchor_generator = build_prior_generator(anchor_generator_cfg)
    assert anchor_generator.num_base_priors == \
           anchor_generator.num_base_anchors
    assert anchor_generator.num_base_priors == [3, 3]
    assert anchor_generator is not None


def test_strides():
    """Test strides."""
    from mmrotate.core import RotatedAnchorGenerator
    # Square strides
    self = RotatedAnchorGenerator([10], [1.], [1.], [10])
    anchors = self.grid_priors([(2, 2)], device='cpu')

    expected_anchors = torch.tensor([[0., 0., 10., 10., 0.],
                                     [10., 0., 10., 10., 0.],
                                     [0., 10., 10., 10., 0.],
                                     [10., 10., 10., 10., 0.]])

    assert torch.equal(anchors[0], expected_anchors)

    # Different strides in x and y direction
    self = RotatedAnchorGenerator([(10, 20)], [1.], [1.], [10])
    anchors = self.grid_priors([(2, 2)], device='cpu')

    expected_anchors = torch.tensor([[0., 0., 10., 10., 0.],
                                     [10., 0., 10., 10., 0.],
                                     [0., 20., 10., 10., 0.],
                                     [10., 20., 10., 10., 0.]])

    assert torch.equal(anchors[0], expected_anchors)
