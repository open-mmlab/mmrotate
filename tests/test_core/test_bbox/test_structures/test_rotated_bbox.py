# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt
from unittest import TestCase

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmrotate.core.bbox.structures import RotatedBoxes


class TestHorizontalBoxes(TestCase):

    def test_regularize_bboxes(self):
        th_bboxes = torch.rand((3, 4, 5))
        th_bboxes[..., 4] = (th_bboxes[..., 4] - 0.5) * 4 * np.pi
        bboxes = RotatedBoxes(th_bboxes)

        th_bboxes = bboxes.regularize_bboxes(
            width_longer=False, start_angle=-30)
        self.assertTrue(th_bboxes[..., 4].min() >= -np.pi / 6)
        self.assertTrue(th_bboxes[..., 4].max() < -np.pi / 6 + np.pi / 2)
        th_bboxes = bboxes.regularize_bboxes(
            width_longer=True, start_angle=-30)
        self.assertTrue(th_bboxes[..., 4].min() >= -np.pi / 6)
        self.assertTrue(th_bboxes[..., 4].max() < -np.pi / 6 + np.pi)
        self.assertTrue((th_bboxes[..., 2] >= th_bboxes[..., 3]).all())

        # test patterns
        # oc
        th_bboxes = bboxes.regularize_bboxes('oc')
        self.assertTrue(th_bboxes[..., 4].min() >= -np.pi / 2)
        self.assertTrue(th_bboxes[..., 4].max() < 0)
        # le90
        th_bboxes = bboxes.regularize_bboxes('le90')
        self.assertTrue(th_bboxes[..., 4].min() >= -np.pi / 2)
        self.assertTrue(th_bboxes[..., 4].max() < np.pi)
        self.assertTrue((th_bboxes[..., 2] >= th_bboxes[..., 3]).all())
        # le135
        th_bboxes = bboxes.regularize_bboxes('le135')
        self.assertTrue(th_bboxes[..., 4].min() >= -np.pi / 4)
        self.assertTrue(th_bboxes[..., 4].max() < 3 * np.pi / 4)
        self.assertTrue((th_bboxes[..., 2] >= th_bboxes[..., 3]).all())

    def test_propoerty(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        bboxes = RotatedBoxes(th_bboxes)

        # Centers
        centers = torch.Tensor([40, 40]).reshape(1, 1, 2)
        assert_allclose(bboxes.centers, centers)
        # Areas
        areas = torch.Tensor([200]).reshape(1, 1)
        assert_allclose(bboxes.areas, areas)
        # widths
        widths = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(bboxes.widths, widths)
        # heights
        heights = torch.Tensor([20]).reshape(1, 1)
        assert_allclose(bboxes.heights, heights)

    def test_flip(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        img_shape = [50, 85]
        bboxes = RotatedBoxes(th_bboxes)

        # horizontal flip
        flipped_bboxes_th = torch.Tensor([45, 40, 10, 20,
                                          -np.pi / 6]).reshape(1, 1, 5)
        flipped_bboxes = bboxes.flip(img_shape, direction='horizontal')
        assert_allclose(flipped_bboxes.tensor, flipped_bboxes_th)
        # vertical flip
        flipped_bboxes_th = torch.Tensor([40, 10, 10, 20,
                                          -np.pi / 6]).reshape(1, 1, 5)
        flipped_bboxes = bboxes.flip(img_shape, direction='vertical')
        assert_allclose(flipped_bboxes.tensor, flipped_bboxes_th)
        # diagonal flip
        flipped_bboxes_th = torch.Tensor([45, 10, 10, 20,
                                          np.pi / 6]).reshape(1, 1, 5)
        flipped_bboxes = bboxes.flip(img_shape, direction='diagonal')
        assert_allclose(flipped_bboxes.tensor, flipped_bboxes_th)

    def test_translate(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        bboxes = RotatedBoxes(th_bboxes)

        translated_bboxes = bboxes.translate([23, 46])
        translated_bboxes_th = torch.Tensor([63, 86, 10, 20,
                                             np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(translated_bboxes.tensor, translated_bboxes_th)
        # negative
        translated_bboxes = bboxes.translate([-6, -2])
        translated_bboxes_th = torch.Tensor([34, 38, 10, 20,
                                             np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(translated_bboxes.tensor, translated_bboxes_th)

    def test_clip(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        img_shape = [13, 14]
        bboxes = RotatedBoxes(th_bboxes)

        cliped_bboxes = bboxes.clip(img_shape)
        cliped_bboxes_th = torch.Tensor([40, 40, 10, 20,
                                         np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(cliped_bboxes.tensor, cliped_bboxes_th)
        # test clone
        self.assertIsNot(cliped_bboxes.tensor, th_bboxes)

    def test_rotate(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        center = (50, 40)
        angle = 60
        bboxes = RotatedBoxes(th_bboxes)

        rotated_bboxes = bboxes.rotate(center, angle)
        rotated_bboxes_th = torch.Tensor(
            [45, 40 + 5 * sqrt(3), 10, 20, -np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(rotated_bboxes.tensor, rotated_bboxes_th)

    def test_project(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        matrix = np.random.rand(3, 3)
        bboxes = RotatedBoxes(th_bboxes)
        bboxes.project(matrix)

    def test_rescale(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        scale_factor = [0.4, 0.4]
        bboxes = RotatedBoxes(th_bboxes)

        rescaled_bboxes = bboxes.rescale(scale_factor)
        rescaled_bboxes_th = torch.Tensor([16, 16, 4, 8,
                                           np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(rescaled_bboxes.tensor, rescaled_bboxes_th)
        rescaled_bboxes = bboxes.rescale(scale_factor, mapping_back=True)
        rescaled_bboxes_th = torch.Tensor([100, 100, 25, 50,
                                           np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(rescaled_bboxes.tensor, rescaled_bboxes_th)

    def test_resize_bboxes(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        scale_factor = [0.4, 0.8]
        bboxes = RotatedBoxes(th_bboxes)

        resized_bboxes = bboxes.resize_bboxes(scale_factor)
        resized_bboxes_th = torch.Tensor([40, 40, 4, 16,
                                          np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(resized_bboxes.tensor, resized_bboxes_th)

    def test_is_bboxes_inside(self):
        th_bboxes = torch.Tensor([[10, 10, 10, 20, 0.3], [25, 25, 10, 20, 0.2],
                                  [35, 35, 10, 20, 0.1]]).reshape(1, 3, 5)
        img_shape = [30, 30]
        bboxes = RotatedBoxes(th_bboxes)

        index = bboxes.is_bboxes_inside(img_shape)
        index_th = torch.BoolTensor([True, True, False]).reshape(1, 3)
        self.assertEqual(tuple(index.size()), (1, 3))
        assert_allclose(index, index_th)

    def test_find_inside_points(self):
        th_bboxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        bboxes = RotatedBoxes(th_bboxes)
        points = torch.Tensor([[20, 40], [30, 40], [37.5, 40], [40, 40]])
        index = bboxes.find_inside_points(points)
        index_th = torch.BoolTensor([False, False, True, True]).reshape(4, 1)
        self.assertEqual(tuple(index.size()), (4, 1))
        assert_allclose(index, index_th)
