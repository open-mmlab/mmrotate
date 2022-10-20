# Copyright (c) OpenMMLab. All rights reserved.
import random
from math import sqrt
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmengine.testing import assert_allclose

from mmrotate.structures.bbox import QuadriBoxes


class TestQuadriBoxes(TestCase):

    def test_propoerty(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                 20]).reshape(1, 1, 8)
        boxes = QuadriBoxes(th_boxes)

        # Centers
        centers = torch.Tensor([17, 15]).reshape(1, 1, 2)
        assert_allclose(boxes.centers, centers)
        # Areas
        areas = torch.Tensor([100]).reshape(1, 1)
        assert_allclose(boxes.areas, areas)
        # widths
        widths = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(boxes.widths, widths)
        # heights
        heights = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(boxes.heights, heights)

    def test_flip(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                 20]).reshape(1, 1, 8)
        img_shape = [50, 85]

        # horizontal flip
        boxes = QuadriBoxes(th_boxes)
        flipped_boxes_th = torch.Tensor([75, 10, 65, 10, 61, 20, 71,
                                         20]).reshape(1, 1, 8)
        boxes.flip_(img_shape, direction='horizontal')
        assert_allclose(boxes.tensor, flipped_boxes_th)
        # vertical flip
        boxes = QuadriBoxes(th_boxes)
        flipped_boxes_th = torch.Tensor([10, 40, 20, 40, 24, 30, 14,
                                         30]).reshape(1, 1, 8)
        boxes.flip_(img_shape, direction='vertical')
        assert_allclose(boxes.tensor, flipped_boxes_th)
        # diagonal flip
        boxes = QuadriBoxes(th_boxes)
        flipped_boxes_th = torch.Tensor([75, 40, 65, 40, 61, 30, 71,
                                         30]).reshape(1, 1, 8)
        boxes.flip_(img_shape, direction='diagonal')
        assert_allclose(boxes.tensor, flipped_boxes_th)

    def test_translate(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                 20]).reshape(1, 1, 8)
        boxes = QuadriBoxes(th_boxes)
        boxes.translate_([23, 46])
        translated_boxes_th = torch.Tensor([33, 56, 43, 56, 47, 66, 37,
                                            66]).reshape(1, 1, 8)
        assert_allclose(boxes.tensor, translated_boxes_th)

    def test_clip(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                 20]).reshape(1, 1, 8)
        img_shape = [13, 14]
        boxes = QuadriBoxes(th_boxes)
        boxes.clip_(img_shape)
        cliped_boxes_th = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                        20]).reshape(1, 1, 8)
        assert_allclose(boxes.tensor, cliped_boxes_th)

    def test_rotate(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 20, 20, 10,
                                 20]).reshape(1, 1, 8)
        center = (15, 15)
        angle = -45
        boxes = QuadriBoxes(th_boxes)
        boxes.rotate_(center, angle)
        rotated_boxes_th = torch.Tensor([
            15 - 5 * sqrt(2), 15, 15, 15 - 5 * sqrt(2), 15 + 5 * sqrt(2), 15,
            15, 15 + 5 * sqrt(2)
        ]).reshape(1, 1, 8)
        assert_allclose(boxes.tensor, rotated_boxes_th)

    def test_project(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                 20]).reshape(1, 1, 8)
        boxes1 = QuadriBoxes(th_boxes)
        boxes2 = boxes1.clone()

        matrix = np.zeros((3, 3), dtype=np.float32)
        center = [random.random() * 80, random.random() * 80]
        angle = random.random() * 180
        matrix[:2, :3] = cv2.getRotationMatrix2D(center, angle, 1)
        x_translate = random.random() * 40
        y_translate = random.random() * 40
        matrix[0, 2] = matrix[0, 2] + x_translate
        matrix[1, 2] = matrix[1, 2] + y_translate
        scale_factor = random.random() * 2
        matrix[2, 2] = 1 / scale_factor
        boxes1.project_(matrix)

        boxes2.rotate_(center, -angle)
        boxes2.translate_([x_translate, y_translate])
        boxes2.rescale_([scale_factor, scale_factor])
        assert_allclose(boxes1.tensor, boxes2.tensor)

    def test_rescale(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                 20]).reshape(1, 1, 8)
        scale_factor = [0.4, 0.8]

        boxes = QuadriBoxes(th_boxes)
        boxes.rescale_(scale_factor)
        rescaled_boxes_th = torch.Tensor([4, 8, 8, 8, 9.6, 16, 5.6,
                                          16]).reshape(1, 1, 8)
        assert_allclose(boxes.tensor, rescaled_boxes_th)

    def test_resize(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14,
                                 20]).reshape(1, 1, 8)
        boxes = QuadriBoxes(th_boxes)
        boxes.resize_([0.4, 0.4])
        resized_boxes_th = torch.Tensor(
            [14.2, 13, 18.2, 13, 19.8, 17, 15.8, 17]).reshape(1, 1, 8)
        assert_allclose(boxes.tensor, resized_boxes_th)

    def test_is_inside(self):
        th_boxes = torch.Tensor([[10, 10, 20, 10, 24, 20, 14, 20],
                                 [20, 10, 30, 10, 34, 20, 24, 20],
                                 [25, 10, 35, 10, 39, 20, 29,
                                  20]]).reshape(1, 3, 8)
        img_shape = [30, 30]
        boxes = QuadriBoxes(th_boxes)

        index = boxes.is_inside(img_shape)
        index_th = torch.BoolTensor([True, True, False]).reshape(1, 3)
        assert_allclose(index, index_th)

    def test_find_inside_points(self):
        th_boxes = torch.Tensor([10, 10, 20, 10, 24, 20, 14, 20]).reshape(1, 8)
        boxes = QuadriBoxes(th_boxes)
        points = torch.Tensor([[9, 15], [11, 15], [12.5, 15], [17, 15]])
        index = boxes.find_inside_points(points)
        index_th = torch.BoolTensor([False, False, True, True]).reshape(4, 1)
        assert_allclose(index, index_th)
        # test is_aligned
        boxes = boxes.expand(4, 8)
        index = boxes.find_inside_points(points, is_aligned=True)
        index_th = torch.BoolTensor([False, False, True, True])
        assert_allclose(index, index_th)

    def test_from_instance_masks(self):
        bitmap_masks = BitmapMasks.random()
        boxes = QuadriBoxes.from_instance_masks(bitmap_masks)
        self.assertIsInstance(boxes, QuadriBoxes)
        self.assertEqual(len(boxes), len(bitmap_masks))
        polygon_masks = PolygonMasks.random()
        boxes = QuadriBoxes.from_instance_masks(polygon_masks)
        self.assertIsInstance(boxes, QuadriBoxes)
        self.assertEqual(len(boxes), len(bitmap_masks))
        # zero length masks
        bitmap_masks = BitmapMasks.random(num_masks=0)
        boxes = QuadriBoxes.from_instance_masks(bitmap_masks)
        self.assertIsInstance(boxes, QuadriBoxes)
        self.assertEqual(len(boxes), 0)
        polygon_masks = PolygonMasks.random(num_masks=0)
        boxes = QuadriBoxes.from_instance_masks(polygon_masks)
        self.assertIsInstance(boxes, QuadriBoxes)
        self.assertEqual(len(boxes), 0)
