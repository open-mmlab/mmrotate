# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.testing import assert_allclose

from mmrotate.models.task_modules.coders import CSLCoder


class TestCSLCoder(TestCase):

    def test_encode(self):
        coder = CSLCoder(angle_version='oc', omega=10)
        angle_preds = torch.Tensor([[0.]])
        expected_encode_angles = torch.Tensor(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        out = coder.encode(angle_preds)
        assert_allclose(expected_encode_angles, out)

    def test_decode(self):
        coder = CSLCoder(angle_version='oc', omega=10)
        encode_angles = torch.Tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        out = coder.decode(encode_angles)
        assert_allclose([1], out.shape)
