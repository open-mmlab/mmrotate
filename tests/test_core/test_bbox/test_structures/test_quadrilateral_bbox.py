# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt
from unittest import TestCase

import numpy as np
import torch
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmengine.testing import assert_allclose

from mmrotate.core.bbox.structures import QuadriBoxes


class TestQuadriBoxes(TestCase):

    def test_propoerty(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                  20]).reshape(1, 1, 8)
        bboxes = QuadriBoxes(th_bboxes)

        # Centers
        centers = torch.Tensor([17, 15]).reshape(1, 1, 2)
        assert_allclose(bboxes.centers, centers)
        # Areas
        areas = torch.Tensor([100]).reshape(1, 1)
        assert_allclose(bboxes.areas, areas)
        # widths
        widths = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(bboxes.widths, widths)
        # heights
        heights = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(bboxes.heights, heights)

    def test_flip(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                  20]).reshape(1, 1, 8)
        img_shape = [50, 85]

        # horizontal flip
        bboxes = QuadriBoxes(th_bboxes)
        flipped_bboxes_th = torch.Tensor([75, 10, 65, 10, 61, 20, 71,
                                          20]).reshape(1, 1, 8)
        bboxes.flip_(img_shape, direction='horizontal')
        assert_allclose(bboxes.tensor, flipped_bboxes_th)
        # vertical flip
        bboxes = QuadriBoxes(th_bboxes)
        flipped_bboxes_th = torch.Tensor([10, 40, 20, 40, 24, 30, 14,
                                          30]).reshape(1, 1, 8)
        bboxes.flip_(img_shape, direction='vertical')
        assert_allclose(bboxes.tensor, flipped_bboxes_th)
        # diagonal flip
        bboxes = QuadriBoxes(th_bboxes)
        flipped_bboxes_th = torch.Tensor([75, 40, 65, 40, 61, 30, 71,
                                          30]).reshape(1, 1, 8)
        bboxes.flip_(img_shape, direction='diagonal')
        assert_allclose(bboxes.tensor, flipped_bboxes_th)

    def test_translate(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                  20]).reshape(1, 1, 8)
        bboxes = QuadriBoxes(th_bboxes)
        bboxes.translate_([23, 46])
        translated_bboxes_th = torch.Tensor([33, 56, 43, 56, 47, 66, 37,
                                             66]).reshape(1, 1, 8)
        assert_allclose(bboxes.tensor, translated_bboxes_th)

    def test_clip(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                  20]).reshape(1, 1, 8)
        img_shape = [13, 14]
        bboxes = QuadriBoxes(th_bboxes)
        bboxes.clip_(img_shape)
        cliped_bboxes_th = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                         20]).reshape(1, 1, 8)
        assert_allclose(bboxes.tensor, cliped_bboxes_th)

    def test_rotate(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 20, 20, 10,
                                  20]).reshape(1, 1, 8)
        center = (15, 15)
        angle = 45
        bboxes = QuadriBoxes(th_bboxes)
        bboxes.rotate_(center, angle)
        rotated_bboxes_th = torch.Tensor([
            15 - 5 * sqrt(2), 15, 15, 15 - 5 * sqrt(2), 15 + 5 * sqrt(2), 15,
            15, 15 + 5 * sqrt(2)
        ]).reshape(1, 1, 8)
        assert_allclose(bboxes.tensor, rotated_bboxes_th)

    def test_project(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                  20]).reshape(1, 1, 8)
        matrix = np.random.rand(3, 3)
        bboxes = QuadriBoxes(th_bboxes)
        bboxes.project_(matrix)

    def test_rescale(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                  20]).reshape(1, 1, 8)
        scale_factor = [0.4, 0.8]

        bboxes = QuadriBoxes(th_bboxes)
        bboxes.rescale_(scale_factor)
        rescaled_bboxes_th = torch.Tensor([4, 8, 8, 8, 9.6, 16, 5.6,
                                           16]).reshape(1, 1, 8)
        assert_allclose(bboxes.tensor, rescaled_bboxes_th)

    def test_resize(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                  20]).reshape(1, 1, 8)
        bboxes = QuadriBoxes(th_bboxes)
        bboxes.resize_([0.4, 0.4])
        resized_bboxes_th = torch.Tensor(
            [14.2, 13, 18.2, 13, 19.8, 17, 15.8, 17]).reshape(1, 1, 8)
        assert_allclose(bboxes.tensor, resized_bboxes_th)

    def test_is_bboxes_inside(self):
        th_bboxes = torch.Tensor([[10, 10, 20, 10, 24, 20, 14, 20],
                                  [20, 10, 30, 10, 34, 20, 24, 20],
                                  [25, 10, 35, 10, 39, 20, 29,
                                   20]]).reshape(1, 3, 8)
        img_shape = [30, 30]
        bboxes = QuadriBoxes(th_bboxes)

        index = bboxes.is_bboxes_inside(img_shape)
        index_th = torch.BoolTensor([True, True, False]).reshape(1, 3)
        assert_allclose(index, index_th)

    def test_find_inside_points(self):
        th_bboxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                  20]).reshape(1, 8)
        bboxes = QuadriBoxes(th_bboxes)
        points = torch.Tensor([[9, 15], [11, 15], [12.5, 15], [17, 15]])
        index = bboxes.find_inside_points(points)
        index_th = torch.BoolTensor([False, False, True, True]).reshape(4, 1)
        assert_allclose(index, index_th)
        # test is_aligned
        bboxes = bboxes.expand(4, 8)
        index = bboxes.find_inside_points(points, is_aligned=True)
        index_th = torch.BoolTensor([False, False, True, True])
        assert_allclose(index, index_th)

    def test_from_masks(self):
        bitmap_masks = BitmapMasks.random()
        bboxes = QuadriBoxes.from_bitmap_masks(bitmap_masks)
        self.assertIsInstance(bboxes, QuadriBoxes)
        self.assertEqual(len(bboxes), len(bitmap_masks))
        polygon_masks = PolygonMasks.random()
        bboxes = QuadriBoxes.from_polygon_masks(polygon_masks)
        self.assertIsInstance(bboxes, QuadriBoxes)
        self.assertEqual(len(bboxes), len(bitmap_masks))
        # zero length masks
        bitmap_masks = BitmapMasks.random(num_masks=0)
        bboxes = QuadriBoxes.from_bitmap_masks(bitmap_masks)
        self.assertIsInstance(bboxes, QuadriBoxes)
        self.assertEqual(len(bboxes), 0)
        polygon_masks = PolygonMasks.random(num_masks=0)
        bboxes = QuadriBoxes.from_polygon_masks(polygon_masks)
        self.assertIsInstance(bboxes, QuadriBoxes)
        self.assertEqual(len(bboxes), 0)
