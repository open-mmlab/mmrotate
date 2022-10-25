# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.testing import assert_allclose

from mmrotate.models.task_modules.coders import DeltaXYWHQBBoxCoder
from mmrotate.structures.bbox import QuadriBoxes


class TestDeltaXYWHQBBoxCoder(TestCase):

    def test_encode(self):
        coder = DeltaXYWHQBBoxCoder()

        proposals = torch.Tensor([[0., 0., 1., 1.], [0., 0., 2., 2.],
                                  [0., 0., 5., 5.], [5., 5., 10., 10.]])
        gt = torch.Tensor([[0., 0., 1., 0., 1., 1., 0., 1.],
                           [0.1, 0., 1.1, 0., 1.1, 1., 0.1, 1.],
                           [0., 0.1, 1., 0.1, 1., 1.1, 0., 1.1],
                           [0.1, 0.1, 1.1, 0.1, 1.1, 1.1, 0.1, 1.1]])

        expected_encode_bboxes = torch.Tensor(
            [[0.0000, 0.0000, 0.0000, 0.0000],
             [-0.2000, -0.2500, -0.6931, -0.6931],
             [-0.4000, -0.3800, -1.6094, -1.6094],
             [-1.3800, -1.3800, -1.6094, -1.6094]])

        out = coder.encode(HorizontalBoxes(proposals), QuadriBoxes(gt))
        assert_allclose(expected_encode_bboxes, out)
