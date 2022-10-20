# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.testing import assert_allclose

from mmrotate.models.task_modules.coders import DeltaXYWHTRBBoxCoder
from mmrotate.structures.bbox import RotatedBoxes


class TestDeltaBboxCoder(TestCase):

    def test_encode(self):
        coder = DeltaXYWHTRBBoxCoder()

        proposals = torch.Tensor([[0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.1],
                                  [0., 0., 1., 1., 0.1], [5., 5., 5., 5.,
                                                          0.3]])
        gt = torch.Tensor([[0.0000, 0.0000, 1.0000, 1.0000, 0.0000],
                           [1.0000, 1.0000, 2.7183, 2.7183, 0.1000],
                           [0.0000, 0.0000, 7.3891, 0.3679, 1.1000],
                           [8.5000, 0.0000, 3.0327, 6.7493, 1.3000]])

        expected_encode_bboxes = torch.Tensor(
            [[0.0000, 0.0000, 0.0000, 0.0000, -1.5708],
             [1.0000, 1.0000, 1.0000, 1.0000, -1.5708],
             [0.0000, 0.0000, -0.9999, 2.0000, -0.5708],
             [0.7000, -1.0000, 0.3000, -0.5000, -0.5708]])

        out = coder.encode(RotatedBoxes(proposals), RotatedBoxes(gt))
        assert_allclose(expected_encode_bboxes, out)

    def test_decode(self):
        coder = DeltaXYWHTRBBoxCoder()

        rois = torch.Tensor([[0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.1],
                             [0., 0., 1., 1., 0.1], [5., 5., 5., 5., 0.3]])
        deltas = torch.Tensor([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 0.],
                               [0., 0., 2., -1., 1.],
                               [0.7, -1.9, -0.5, 0.3, 1.]])
        expected_decode_bboxes = torch.Tensor(
            [[0.0000, 0.0000, 1.0000, 1.0000, 0.0000],
             [1.0000, 1.0000, 2.7183, 2.7183, 0.1000],
             [0.0000, 0.0000, 7.3891, 0.3679, 1.1000],
             [8.5000, 0.0000, 3.0327, 6.7493, 1.3000]])

        out = coder.decode(
            RotatedBoxes(rois), deltas, max_shape=(32, 32)).tensor
        assert_allclose(expected_decode_bboxes, out)
        out = coder.decode(
            RotatedBoxes(rois), deltas, max_shape=torch.Tensor(
                (32, 32))).tensor
        assert_allclose(expected_decode_bboxes, out)

        # batch decode
        batch_rois = rois.unsqueeze(0).repeat(2, 1, 1)
        batch_deltas = deltas.unsqueeze(0).repeat(2, 1, 1)
        batch_out = coder.decode(
            RotatedBoxes(batch_rois), batch_deltas, max_shape=(32, 32))[0]
        assert_allclose(out, batch_out.tensor)

        # empty deltas
        rois = torch.zeros((0, 5))
        deltas = torch.zeros((0, 5))
        out = coder.decode(RotatedBoxes(rois), deltas, max_shape=(32, 32))
        self.assertEqual(rois.shape, out.shape)

        # test add_ctr_clamp
        coder = DeltaXYWHTRBBoxCoder(add_ctr_clamp=True, ctr_clamp=2)

        rois = torch.Tensor([[0., 0., 6., 6., 0.], [0., 0., 1., 1., 0.1],
                             [0., 0., 1., 1., 0.1], [5., 5., 5., 5., 0.3]])
        deltas = torch.Tensor([[1., 1., 2., 2., 0.], [1., 1., 1., 1., 0.],
                               [0., 0., 2., -1., 1.],
                               [0.7, -1.9, -0.5, 0.3, 1.]])
        expected_decode_bboxes = torch.Tensor(
            [[2.0000, 2.0000, 44.3343, 44.3343, 0.0000],
             [1.0000, 1.0000, 2.7183, 2.7183, 0.1000],
             [0.0000, 0.0000, 7.3891, 0.3679, 1.1000],
             [7.0000, 3.0000, 3.0327, 6.7493, 1.3000]])

        out = coder.decode(
            RotatedBoxes(rois), deltas, max_shape=(32, 32)).tensor
        assert_allclose(expected_decode_bboxes, out)
