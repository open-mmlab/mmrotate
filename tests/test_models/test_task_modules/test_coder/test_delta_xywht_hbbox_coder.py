# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.testing import assert_allclose

from mmrotate.models.task_modules.coders import DeltaXYWHTHBBoxCoder
from mmrotate.structures.bbox import RotatedBoxes


class TestDeltaBboxCoder(TestCase):

    def test_encode(self):
        coder = DeltaXYWHTHBBoxCoder()

        proposals = torch.Tensor([[0., 0., 1., 1.], [0., 0., 2., 2.],
                                  [0., 0., 5., 5.], [5., 5., 10., 10.]])
        gt = torch.Tensor([[0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.1],
                           [0., 0., 1., 1., 0.1], [5., 5., 5., 5., 0.3]])

        expected_encode_bboxes = torch.Tensor(
            [[-0.5000, -0.5000, 0.0000, 0.0000, -1.5708],
             [-0.5000, -0.5000, -0.6931, -0.6931, -1.4708],
             [-0.5000, -0.5000, -1.6094, -1.6094, -1.4708],
             [-0.5000, -0.5000, 0.0000, 0.0000, -1.2708]])

        out = coder.encode(HorizontalBoxes(proposals), RotatedBoxes(gt))
        assert_allclose(expected_encode_bboxes, out)

    def test_decode(self):
        coder = DeltaXYWHTHBBoxCoder()

        rois = torch.Tensor([[0., 0., 1., 1.], [0., 0., 2., 2.],
                             [0., 0., 5., 5.], [5., 5., 10., 10.]])
        deltas = torch.Tensor([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 0.],
                               [0., 0., 2., -1., 1.],
                               [0.7, -1.9, -0.5, 0.3, 1.]])
        expected_decode_bboxes = torch.Tensor(
            [[0.5000, 0.5000, 1.0000, 1.0000, 0.0000],
             [3.0000, 3.0000, 5.4366, 5.4366, 0.0000],
             [2.5000, 2.5000, 36.9453, 1.8394, 1.0000],
             [11.0000, -2.0000, 3.0327, 6.7493, 1.0000]])

        out = coder.decode(
            HorizontalBoxes(rois), deltas, max_shape=(32, 32)).tensor
        assert_allclose(expected_decode_bboxes, out)
        out = coder.decode(
            HorizontalBoxes(rois), deltas, max_shape=torch.Tensor(
                (32, 32))).tensor
        assert_allclose(expected_decode_bboxes, out)

        # batch decode
        batch_rois = rois.unsqueeze(0).repeat(2, 1, 1)
        batch_deltas = deltas.unsqueeze(0).repeat(2, 1, 1)
        batch_out = coder.decode(
            HorizontalBoxes(batch_rois), batch_deltas, max_shape=(32, 32))[0]
        assert_allclose(out, batch_out.tensor)

        # empty deltas
        rois = torch.zeros((0, 4))
        deltas = torch.zeros((0, 5))
        out = coder.decode(HorizontalBoxes(rois), deltas, max_shape=(32, 32))
        self.assertEqual((0, 5), out.shape)

        # test add_ctr_clamp
        coder = DeltaXYWHTHBBoxCoder(add_ctr_clamp=True, ctr_clamp=2)

        rois = torch.Tensor([[0., 0., 6., 6.], [0., 0., 1., 1.],
                             [0., 0., 5., 5.], [5., 5., 10., 10.]])
        deltas = torch.Tensor([[1., 1., 2., 2., 0.], [1., 1., 1., 1., 0.],
                               [0., 0., 2., -1., 1.],
                               [0.7, -1.9, -0.5, 0.3, 1.]])
        expected_decode_bboxes = torch.Tensor(
            [[5.0000, 5.0000, 44.3343, 44.3343, 0.0000],
             [1.5000, 1.5000, 2.7183, 2.7183, 0.0000],
             [2.5000, 2.5000, 36.9453, 1.8394, 1.0000],
             [9.5000, 5.5000, 3.0327, 6.7493, 1.0000]])

        out = coder.decode(
            HorizontalBoxes(rois), deltas, max_shape=(32, 32)).tensor
        assert_allclose(expected_decode_bboxes, out)
