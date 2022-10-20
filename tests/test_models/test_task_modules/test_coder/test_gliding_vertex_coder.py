# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.testing import assert_allclose

from mmrotate.models.task_modules.coders import GVFixCoder, GVRatioCoder
from mmrotate.structures.bbox import QuadriBoxes


class TestGVFixCoder(TestCase):

    def test_encode(self):
        coder = GVFixCoder()

        proposals = torch.Tensor([[0., 0., 1., 0., 1., 1., 0., 1.],
                                  [0.1, 0., 1.1, 0., 1.1, 1., 0.1, 1.],
                                  [0., 0.1, 1., 0.1, 1., 1.1, 0., 1.1],
                                  [0.1, 0.1, 1.1, 0.1, 1.1, 1.1, 0.1, 1.1]])

        expected_encode_bboxes = torch.Tensor([[1., 1., 1., 1.],
                                               [1., 1., 1., 1.],
                                               [1., 1., 1., 1.],
                                               [1., 1., 1., 1.]])

        out = coder.encode(QuadriBoxes(proposals))
        assert_allclose(expected_encode_bboxes, out)

    def test_decode(self):
        coder = GVFixCoder()

        rois = torch.Tensor([[0., 0., 1., 1.], [0., 0., 1., 1.],
                             [0., 0., 1., 1.], [5., 5., 5., 5.]])
        deltas = torch.Tensor([[0., 0., 0., 0.], [1., 1., 1., 1.],
                               [0., 0., 2., -1.], [0.7, -1.9, -0.5, 0.3]])
        expected_decode_bboxes = torch.Tensor(
            [[0., 0., 1., 0., 1., 1., 0.,
              1.], [1., 0., 1., 1., 0., 1., 0., 0.],
             [0., 0., 1., 0., -1., 1., 0., 2.],
             [5., 5., 5., 5., 5., 5., 5., 5.]])

        out = coder.decode(HorizontalBoxes(rois), deltas)
        assert_allclose(expected_decode_bboxes, out)
        out = coder.decode(HorizontalBoxes(rois), deltas)
        assert_allclose(expected_decode_bboxes, out)

        # empty deltas
        rois = torch.zeros((0, 4))
        deltas = torch.zeros((0, 4))
        out = coder.decode(HorizontalBoxes(rois), deltas)
        self.assertEqual(rois.shape, torch.Size([0, 4]))


class TestGVRatioCoder(TestCase):

    def test_encode(self):
        coder = GVRatioCoder()

        proposals = torch.Tensor([[0., 0., 1., 0., 1., 1., 0., 1.],
                                  [0.1, 0., 1.1, 0., 1.1, 1., 0.1, 1.],
                                  [0., 0.1, 1., 0.1, 1., 1.1, 0., 1.1],
                                  [0.1, 0.1, 1.1, 0.1, 1.1, 1.1, 0.1, 1.1]])

        expected_encode_bboxes = torch.Tensor([[1.], [1.], [1.], [1.]])

        out = coder.encode(QuadriBoxes(proposals))
        assert_allclose(expected_encode_bboxes, out)
