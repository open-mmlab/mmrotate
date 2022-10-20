# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt
from unittest import TestCase

import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.testing import assert_allclose

from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes


class TestboxModeConverters(TestCase):

    def test_hbox2rbox(self):
        th_hboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        hboxes = HorizontalBoxes(th_hboxes)

        rboxes = hboxes.convert_to('rbox')
        th_rboxes = torch.Tensor([15, 15, 10, 10, 0]).reshape(1, 1, 5)
        assert_allclose(rboxes.tensor, th_rboxes)

    def test_hbox2qbox(self):
        th_hboxes = torch.Tensor([10, 10, 20, 20]).reshape(1, 1, 4)
        hboxes = HorizontalBoxes(th_hboxes)

        rboxes = hboxes.convert_to('qbox')
        th_qboxes = torch.Tensor([10, 10, 20, 10, 20, 20, 10,
                                  20]).reshape(1, 1, 8)
        assert_allclose(rboxes.tensor, th_qboxes)

    def test_rbox2hbox(self):
        th_rboxes = torch.Tensor([10, 10, 4, 4, np.pi / 4]).reshape(1, 1, 5)
        rboxes = RotatedBoxes(th_rboxes)

        hboxes = rboxes.convert_to('hbox')
        th_hboxes = torch.Tensor([
            10 - 2 * sqrt(2), 10 - 2 * sqrt(2), 10 + 2 * sqrt(2),
            10 + 2 * sqrt(2)
        ]).reshape(1, 1, 4)
        assert_allclose(hboxes.tensor, th_hboxes)

    def test_rbox2qbox(self):
        th_rboxes = torch.Tensor([10, 10, 4, 4, np.pi / 4]).reshape(1, 1, 5)
        rboxes = RotatedBoxes(th_rboxes)

        qboxes = rboxes.convert_to('qbox')
        th_qboxes = torch.Tensor([
            10, 10 + 2 * sqrt(2), 10 + 2 * sqrt(2), 10, 10, 10 - 2 * sqrt(2),
            10 - 2 * sqrt(2), 10
        ]).reshape(1, 1, 8)
        assert_allclose(qboxes.tensor, th_qboxes)

    def test_qbox2hbox(self):
        th_qboxes = torch.Tensor([
            10, 10 + 2 * sqrt(2), 10 + 2 * sqrt(2), 10, 10, 10 - 2 * sqrt(2),
            10 - 2 * sqrt(2), 10
        ]).reshape(1, 1, 8)
        qboxes = QuadriBoxes(th_qboxes)

        hboxes = qboxes.convert_to('hbox')
        th_hboxes = torch.Tensor([
            10 - 2 * sqrt(2), 10 - 2 * sqrt(2), 10 + 2 * sqrt(2),
            10 + 2 * sqrt(2)
        ]).reshape(1, 1, 4)
        assert_allclose(hboxes.tensor, th_hboxes)

    def test_qbox2rbox(self):
        th_qboxes = torch.Tensor([
            10, 10 + 2 * sqrt(2), 10 + 2 * sqrt(2), 10, 10, 10 - 2 * sqrt(2),
            10 - 2 * sqrt(2), 10
        ]).reshape(1, 1, 8)
        qboxes = QuadriBoxes(th_qboxes)

        rboxes = qboxes.convert_to('rbox')
        rboxes.regularize_boxes(width_longer=False, start_angle=0)
        th_rboxes = torch.Tensor([10, 10, 4, 4, np.pi / 4]).reshape(1, 1, 5)
        assert_allclose(rboxes.tensor, th_rboxes)
