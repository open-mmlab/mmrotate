# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample
from parameterized import parameterized

from mmrotate.apis import inference_detector_by_patches
from mmrotate.utils import register_all_modules


class TestInferenceDetectorByPatches(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand([
        ('rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py',
         ('cuda', )),
    ])
    def test_inference_detector_by_patches(self, config, devices):
        assert all([device in ['cpu', 'cuda'] for device in devices])

        project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        project_dir = osp.join(project_dir, '..')

        config_file = osp.join(project_dir, 'configs', config)

        # test init_detector with config_file: str and cfg_options
        rng = np.random.RandomState(0)
        img = rng.randint(0, 255, (125, 125, 3), dtype=np.uint8)

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                pytest.skip('test requires GPU and torch+cuda')

            model = init_detector(config_file, device=device)
            nms_cfg = dict(type='nms_rotated', iou_threshold=0.1)
            result = inference_detector_by_patches(model, img, [75], [50],
                                                   [1.0], nms_cfg)
            assert isinstance(result, DetDataSample)
