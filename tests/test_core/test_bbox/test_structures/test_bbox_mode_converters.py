# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt
from unittest import TestCase

import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.testing import assert_allclose

from mmrotate.core.bbox.structures import QuadriBoxes, RotatedBoxes


class TestBBoxModeConverters(TestCase):

    def test_hbbox2rbbox(self):
        th_hbboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        hbboxes = HorizontalBoxes(th_hbboxes)

        rbboxes = hbboxes.convert_to('rbox')
        th_rbboxes = torch.Tensor([15, 15, 10, 10, 0]).reshape(1, 1, 5)
        assert_allclose(rbboxes.tensor, th_rbboxes)

    def test_hbbox2qbbox(self):
        th_hbboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        hbboxes = HorizontalBoxes(th_hbboxes)

        rbboxes = hbboxes.convert_to('qbox')
        th_qbboxes = torch.Tensor([10, 10, 20, 10, 20, 20, 10,
                                   20]).reshape(1, 1, 8)
        assert_allclose(rbboxes.tensor, th_qbboxes)

    def test_rbbox2hbbox(self):
        th_rbboxes = torch.Tensor([10, 10, 4, 4, np.pi / 4]).reshape(1, 1, 5)
        rbboxes = RotatedBoxes(th_rbboxes)

        hbboxes = rbboxes.convert_to('hbox')
        th_hbboxes = torch.Tensor([
            10 - 2 * sqrt(2), 10 - 2 * sqrt(2), 10 + 2 * sqrt(2),
            10 + 2 * sqrt(2)
        ]).reshape(1, 1, 4)
        assert_allclose(hbboxes.tensor, th_hbboxes)

    def test_rbbox2qbbox(self):
        th_rbboxes = torch.Tensor([10, 10, 4, 4, np.pi / 4]).reshape(1, 1, 5)
        rbboxes = RotatedBoxes(th_rbboxes)

        qbboxes = rbboxes.convert_to('qbox')
        th_qbboxes = torch.Tensor([
            10, 10 + 2 * sqrt(2), 10 + 2 * sqrt(2), 10, 10, 10 - 2 * sqrt(2),
            10 - 2 * sqrt(2), 10
        ]).reshape(1, 1, 8)
        assert_allclose(qbboxes.tensor, th_qbboxes)

    def test_qbbox2hbbox(self):
        th_qbboxes = torch.Tensor([
            10, 10 + 2 * sqrt(2), 10 + 2 * sqrt(2), 10, 10, 10 - 2 * sqrt(2),
            10 - 2 * sqrt(2), 10
        ]).reshape(1, 1, 8)
        qbboxes = QuadriBoxes(th_qbboxes)

        hbboxes = qbboxes.convert_to('hbox')
        th_hbboxes = torch.Tensor([
            10 - 2 * sqrt(2), 10 - 2 * sqrt(2), 10 + 2 * sqrt(2),
            10 + 2 * sqrt(2)
        ]).reshape(1, 1, 4)
        assert_allclose(hbboxes.tensor, th_hbboxes)

    def test_qbbox2rbbox(self):
        th_qbboxes = torch.Tensor([
            10, 10 + 2 * sqrt(2), 10 + 2 * sqrt(2), 10, 10, 10 - 2 * sqrt(2),
            10 - 2 * sqrt(2), 10
        ]).reshape(1, 1, 8)
        qbboxes = QuadriBoxes(th_qbboxes)

        rbboxes = qbboxes.convert_to('rbox')
        rbboxes.regularize_bboxes(width_longer=False, start_angle=0)
        th_rbboxes = torch.Tensor([10, 10, 4, 4, np.pi / 4]).reshape(1, 1, 5)
        assert_allclose(rbboxes.tensor, th_rbboxes)
