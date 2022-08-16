# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt
from unittest import TestCase
import numpy as np
import torch

from mmrotate.models.data_preprocessors import RotDataPreprocessor
from mmengine.testing import assert_allclose
from mmdet.testing import demo_mm_inputs

class TestRotDataPreprocessor(TestCase):

    def test_init(self):
        # test angle_version is error
        with self.assertRaises(AssertionError):
            RotDataPreprocessor(angle_version='le60')

    def test_forward(self):
        packed_inputs = demo_mm_inputs(2, [[3, 128, 128]])
        packed_inputs[0]['data_sample'].gt_instances.bboxes = torch.Tensor([
            10, 10 + 2 * sqrt(2), 10 + 2 * sqrt(2), 10, 10, 10 - 2 * sqrt(2),
            10 - 2 * sqrt(2), 10
        ]).reshape(1, 1, 8)
        processor = RotDataPreprocessor(angle_version='le90')
        inputs, data_samples = processor(packed_inputs)        

        th_rboxes = torch.Tensor([10, 10, 4, 4, np.pi / 4]).reshape(1, 1, 5)
        assert_allclose(data_samples.gt_instances.bboxes.tensor, th_rboxes)
 