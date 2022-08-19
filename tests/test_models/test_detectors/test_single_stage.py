# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest
from unittest import TestCase

import torch
<<<<<<< HEAD
<<<<<<< HEAD
from mmdet.structures import DetDataSample
from mmengine.logging import MessageHub
from parameterized import parameterized

=======
from mmengine.logging import MessageHub
from parameterized import parameterized

from mmdet.structures import DetDataSample
>>>>>>> 61dcdf7 (init)
=======
from mmdet.structures import DetDataSample
from mmengine.logging import MessageHub
from parameterized import parameterized

>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
from mmrotate.testing import demo_mm_inputs, get_detector_cfg
from mmrotate.utils import register_all_modules


class TestSingleStageDetector(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand([
        'rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
    ])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmrotate.models import build_detector
        detector = build_detector(model)
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.bbox_head)

    @parameterized.expand([
<<<<<<< HEAD
<<<<<<< HEAD
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
         ('cpu', 'cuda')),
=======
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py', ('cpu', 'cuda')),
>>>>>>> 61dcdf7 (init)
=======
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
         ('cpu', 'cuda')),
>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
    ])
    def test_single_stage_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_single_stage_forward_loss_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmrotate.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)
            detector.init_weights()

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c95ac9c (fix lint)
            packed_inputs = demo_mm_inputs(
                2, [[3, 128, 128], [3, 125, 130]], with_boxlist=True)
            data = detector.data_preprocessor(packed_inputs, True)
            losses = detector.forward(**data, mode='loss')
<<<<<<< HEAD
            self.assertIsInstance(losses, dict)

    @parameterized.expand([
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
         ('cpu', 'cuda')),
=======

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
            batch_inputs, data_samples = detector.data_preprocessor(
                packed_inputs, True)
            losses = detector.forward(batch_inputs, data_samples, mode='loss')
=======
>>>>>>> c95ac9c (fix lint)
            self.assertIsInstance(losses, dict)

    @parameterized.expand([
<<<<<<< HEAD
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py', ('cpu', 'cuda')),
>>>>>>> 61dcdf7 (init)
=======
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
         ('cpu', 'cuda')),
>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
    ])
    def test_single_stage_forward_predict_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmrotate.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
<<<<<<< HEAD
<<<<<<< HEAD
            data = detector.data_preprocessor(packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(**data, mode='predict')
=======
            batch_inputs, data_samples = detector.data_preprocessor(
                packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(
                    batch_inputs, data_samples, mode='predict')
>>>>>>> 61dcdf7 (init)
=======
            data = detector.data_preprocessor(packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(**data, mode='predict')
>>>>>>> c95ac9c (fix lint)
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([
<<<<<<< HEAD
<<<<<<< HEAD
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
         ('cpu', 'cuda')),
=======
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py', ('cpu', 'cuda')),
>>>>>>> 61dcdf7 (init)
=======
        ('rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
         ('cpu', 'cuda')),
>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
    ])
    def test_single_stage_forward_tensor_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmrotate.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
<<<<<<< HEAD
<<<<<<< HEAD
            data = detector.data_preprocessor(packed_inputs, False)
            batch_results = detector.forward(**data, mode='tensor')
=======
            batch_inputs, data_samples = detector.data_preprocessor(
                packed_inputs, False)
            batch_results = detector.forward(
                batch_inputs, data_samples, mode='tensor')
>>>>>>> 61dcdf7 (init)
=======
            data = detector.data_preprocessor(packed_inputs, False)
            batch_results = detector.forward(**data, mode='tensor')
>>>>>>> c95ac9c (fix lint)
            self.assertIsInstance(batch_results, tuple)
