# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.testing import assert_allclose

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules


class TestFakeRotatedAnchorGenerator(TestCase):

    def setUp(self):
        register_all_modules()

    def test_standard_anchor_generator(self):
        anchor_generator_cfg = dict(
            type='FakeRotatedAnchorGenerator',
            angle_version='oc',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[4, 8])

        anchor_generator = TASK_UTILS.build(anchor_generator_cfg)
        self.assertEqual(anchor_generator.num_base_priors,
                         anchor_generator.num_base_anchors)
        self.assertEqual(anchor_generator.num_base_priors, [9, 9])
        self.assertIsNotNone(anchor_generator)

    def test_strides(self):
        from mmrotate.models.task_modules.prior_generators import \
            FakeRotatedAnchorGenerator

        # Square strides
        # self = FakeRotatedAnchorGenerator([10], [1.], [1.], [10])
        faker_anchor_generator = FakeRotatedAnchorGenerator(
            angle_version='oc',
            strides=[10],
            ratios=[1.0],
            scales=[1.0],
            base_sizes=[10])

        anchors = faker_anchor_generator.grid_priors([(2, 2)], device='cpu')[0]
        self.assertIsInstance(anchors, RotatedBoxes)

        expected_anchors = torch.tensor(
            [[0.0000, 0.0000, 10.0000, 10.0000, -1.5708],
             [10.0000, 0.0000, 10.0000, 10.0000, -1.5708],
             [0.0000, 10.0000, 10.0000, 10.0000, -1.5708],
             [10.0000, 10.0000, 10.0000, 10.0000, -1.5708]])

        assert_allclose(anchors.tensor, expected_anchors)


class TestPseudoRotatedAnchorGenerator(TestCase):

    def setUp(self):
        register_all_modules()

    def test_standard_anchor_generator(self):
        anchor_generator_cfg = dict(
            type='PseudoRotatedAnchorGenerator', strides=[8, 16, 32, 64, 128])

        anchor_generator = TASK_UTILS.build(anchor_generator_cfg)
        self.assertEqual(anchor_generator.num_base_priors,
                         anchor_generator.num_base_anchors)
        self.assertEqual(anchor_generator.num_base_priors, [1, 1, 1, 1, 1])
        self.assertIsNotNone(anchor_generator)
