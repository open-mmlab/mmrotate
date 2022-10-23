# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmdet.structures import DetDataSample
from parameterized import parameterized

from mmrotate.testing import demo_mm_inputs, get_detector_cfg
from mmrotate.utils import register_all_modules


class TestTwoStageBBox(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand([
        'rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py',
        'oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py',
    ])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.registry import MODELS
        detector = MODELS.build(model)
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.rpn_head)
        self.assertTrue(detector.roi_head)

        # if rpn.num_classes > 1, force set rpn.num_classes = 1
        if hasattr(model.rpn_head, 'num_classes'):
            model.rpn_head.num_classes = 2
            detector = MODELS.build(model)
            self.assertEqual(detector.rpn_head.num_classes, 1)

    @parameterized.expand([
        'rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py',
        'oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py',
    ])
    def test_two_stage_forward_loss_mode(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.registry import MODELS
        detector = MODELS.build(model)

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.cuda()

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]], use_box_type=True)

        data = detector.data_preprocessor(packed_inputs, True)
        # Test loss mode
        losses = detector.forward(**data, mode='loss')
        self.assertIsInstance(losses, dict)

    @parameterized.expand([
        'rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py',
        'oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py',
    ])
    def test_two_stage_forward_predict_mode(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.registry import MODELS
        detector = MODELS.build(model)

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.cuda()

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]], use_box_type=True)
        data = detector.data_preprocessor(packed_inputs, False)
        # Test forward test
        detector.eval()
        with torch.no_grad():
            with torch.no_grad():
                batch_results = detector.forward(**data, mode='predict')
            self.assertEqual(len(batch_results), 2)
            self.assertIsInstance(batch_results[0], DetDataSample)
