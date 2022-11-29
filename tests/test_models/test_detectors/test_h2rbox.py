# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest
from unittest import TestCase

import torch
from mmdet.structures import DetDataSample
from mmengine.logging import MessageHub
from parameterized import parameterized

from mmrotate.testing import demo_mm_inputs, get_detector_cfg
from mmrotate.utils import register_all_modules


class TestSingleStageDetector(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand([
        'h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota.py',
    ])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmrotate.registry import MODELS
        detector = MODELS.build(model)
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.bbox_head)

    @parameterized.expand([
        ('h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota.py', ('cpu', 'cuda')),
    ])
    def test_single_stage_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_single_stage_forward_loss_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmrotate.registry import MODELS
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = MODELS.build(model)
            detector.init_weights()

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()
            packed_inputs = demo_mm_inputs(
                2, [[3, 128, 128], [3, 125, 130]], use_box_type=True)
            data = detector.data_preprocessor(packed_inputs, True)
            losses = detector.forward(**data, mode='loss')
            self.assertIsInstance(losses, dict)

    @parameterized.expand([
        ('h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota.py', ('cpu', 'cuda')),
    ])
    def test_single_stage_forward_predict_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmrotate.registry import MODELS
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = MODELS.build(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(
                2, [[3, 128, 128], [3, 125, 130]], use_box_type=True)
            data = detector.data_preprocessor(packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(**data, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([
        ('h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota.py', ('cpu', 'cuda')),
    ])
    def test_single_stage_forward_tensor_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmrotate.registry import MODELS
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = MODELS.build(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(
                2, [[3, 128, 128], [3, 125, 130]], use_box_type=True)
            data = detector.data_preprocessor(packed_inputs, False)
            batch_results = detector.forward(**data, mode='tensor')
            self.assertIsInstance(batch_results, tuple)
