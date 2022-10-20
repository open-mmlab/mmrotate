# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.testing import assert_allclose

from mmrotate.models.task_modules.coders import MidpointOffsetCoder
from mmrotate.structures.bbox import RotatedBoxes


class TestDeltaBboxCoder(TestCase):

    def test_encode(self):
        coder = MidpointOffsetCoder()

        proposals = torch.Tensor([[0., 0., 1., 1.], [0., 0., 2., 2.],
                                  [0., 0., 5., 5.], [5., 5., 10., 10.]])
        gt = torch.Tensor([[0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.1],
                           [0., 0., 1., 1., 0.1], [5., 5., 5., 5., 0.3]])

        expected_encode_bboxes = torch.Tensor(
            [[-0.5000, -0.5000, 0.0000, 0.0000, 0.5000, 0.5000],
             [-0.5000, -0.5000, -0.6025, -0.6025, 0.5000, 0.5000],
             [-0.5000, -0.5000, -1.5188, -1.5188, 0.5000, 0.5000],
             [-0.5000, -0.5000, 0.2238, 0.2238, -0.2637, -0.2637]])

        out = coder.encode(HorizontalBoxes(proposals), RotatedBoxes(gt))
        assert_allclose(expected_encode_bboxes, out, atol=1e-03, rtol=1e-04)

    def test_decode(self):
        coder = MidpointOffsetCoder()

        rois = torch.Tensor([[0., 0., 1., 1.], [0., 0., 2., 2.],
                             [0., 0., 5., 5.], [5., 5., 10., 10.]])
        deltas = torch.Tensor([[0., 0., 0., 0., 0., 1.],
                               [1., 1., 1., 1., 0., 0.],
                               [0., 0., 2., -1., 1., 1.],
                               [0.7, -1.9, -0.5, 0.3, 1., 0.]])
        expected_decode_bboxes = torch.Tensor(
            [[0.5000, 0.5000, 0.5412, 1.3066, -0.3927],
             [3.0000, 3.0000, 3.8442, 3.8442, -0.7854],
             [2.5000, 2.5000, 1.8394, 36.9453, -1.5708],
             [11.0000, -2.0000, 6.2125, 4.0194, -0.5743]])

        out = coder.decode(
            HorizontalBoxes(rois), deltas, max_shape=(32, 32)).tensor
        assert_allclose(expected_decode_bboxes, out, atol=1e-03, rtol=1e-04)
        out = coder.decode(
            HorizontalBoxes(rois), deltas, max_shape=torch.Tensor(
                (32, 32))).tensor
        assert_allclose(expected_decode_bboxes, out, atol=1e-03, rtol=1e-04)

        # batch decode
        batch_rois = rois.unsqueeze(0).repeat(2, 1, 1)
        batch_deltas = deltas.unsqueeze(0).repeat(2, 1, 1)
        batch_out = coder.decode(
            HorizontalBoxes(batch_rois), batch_deltas, max_shape=(32, 32))[0]
        assert_allclose(out, batch_out.tensor, atol=1e-03, rtol=1e-04)

        # empty deltas
        rois = torch.zeros((0, 4))
        deltas = torch.zeros((0, 6))
        out = coder.decode(HorizontalBoxes(rois), deltas, max_shape=(32, 32))
        self.assertEqual((0, 5), out.shape)
