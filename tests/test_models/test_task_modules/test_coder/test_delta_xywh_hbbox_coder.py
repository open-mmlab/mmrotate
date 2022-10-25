# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.testing import assert_allclose

from mmrotate.models.task_modules.coders import DeltaXYWHHBBoxCoder
from mmrotate.structures.bbox import RotatedBoxes


class TestDeltaBboxCoder(TestCase):

    def test_encode(self):
        coder = DeltaXYWHHBBoxCoder()

        proposals = torch.Tensor([[0., 0., 1., 1.], [0., 0., 2., 2.],
                                  [0., 0., 5., 5.], [5., 5., 10., 10.]])
        gt = torch.Tensor([[0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.1],
                           [0., 0., 1., 1., 0.1], [5., 5., 5., 5., 0.3]])

        expected_encode_bboxes = torch.Tensor(
            [[-0.5000, -0.5000, 0.0000, 0.0000],
             [-0.5000, -0.5000, -0.6025, -0.6025],
             [-0.5000, -0.5000, -1.5188, -1.5188],
             [-0.5000, -0.5000, 0.2238, 0.2238]])

        out = coder.encode(HorizontalBoxes(proposals), RotatedBoxes(gt))
        assert_allclose(expected_encode_bboxes, out)
