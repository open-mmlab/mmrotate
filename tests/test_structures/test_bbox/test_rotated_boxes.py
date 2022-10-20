# Copyright (c) OpenMMLab. All rights reserved.
import random
from math import sqrt
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmengine.testing import assert_allclose

from mmrotate.structures.bbox import RotatedBoxes


class TestRotatedBoxes(TestCase):

    def test_regularize_boxes(self):
        th_boxes = torch.rand((3, 4, 5))
        th_boxes[..., 4] = (th_boxes[..., 4] - 0.5) * 4 * np.pi
        boxes = RotatedBoxes(th_boxes)

        th_boxes = boxes.regularize_boxes(width_longer=False, start_angle=-30)
        self.assertTrue(th_boxes[..., 4].min() >= -np.pi / 6)
        self.assertTrue(th_boxes[..., 4].max() < -np.pi / 6 + np.pi / 2)
        th_boxes = boxes.regularize_boxes(width_longer=True, start_angle=-30)
        self.assertTrue(th_boxes[..., 4].min() >= -np.pi / 6)
        self.assertTrue(th_boxes[..., 4].max() < -np.pi / 6 + np.pi)
        self.assertTrue((th_boxes[..., 2] >= th_boxes[..., 3]).all())

        # test patterns
        # oc
        th_boxes = boxes.regularize_boxes('oc')
        self.assertTrue(th_boxes[..., 4].min() >= -np.pi / 2)
        self.assertTrue(th_boxes[..., 4].max() < 0)
        # le90
        th_boxes = boxes.regularize_boxes('le90')
        self.assertTrue(th_boxes[..., 4].min() >= -np.pi / 2)
        self.assertTrue(th_boxes[..., 4].max() < np.pi)
        self.assertTrue((th_boxes[..., 2] >= th_boxes[..., 3]).all())
        # le135
        th_boxes = boxes.regularize_boxes('le135')
        self.assertTrue(th_boxes[..., 4].min() >= -np.pi / 4)
        self.assertTrue(th_boxes[..., 4].max() < 3 * np.pi / 4)
        self.assertTrue((th_boxes[..., 2] >= th_boxes[..., 3]).all())

    def test_propoerty(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        boxes = RotatedBoxes(th_boxes)

        # Centers
        centers = torch.Tensor([40, 40]).reshape(1, 1, 2)
        assert_allclose(boxes.centers, centers)
        # Areas
        areas = torch.Tensor([200]).reshape(1, 1)
        assert_allclose(boxes.areas, areas)
        # widths
        widths = torch.Tensor([10]).reshape(1, 1)
        assert_allclose(boxes.widths, widths)
        # heights
        heights = torch.Tensor([20]).reshape(1, 1)
        assert_allclose(boxes.heights, heights)

    def test_flip(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        img_shape = [50, 85]
        # horizontal flip
        boxes = RotatedBoxes(th_boxes)
        flipped_boxes_th = torch.Tensor([45, 40, 10, 20,
                                         -np.pi / 6]).reshape(1, 1, 5)
        boxes.flip_(img_shape, direction='horizontal')
        assert_allclose(boxes.tensor, flipped_boxes_th)
        # vertical flip
        boxes = RotatedBoxes(th_boxes)
        flipped_boxes_th = torch.Tensor([40, 10, 10, 20,
                                         -np.pi / 6]).reshape(1, 1, 5)
        boxes.flip_(img_shape, direction='vertical')
        assert_allclose(boxes.tensor, flipped_boxes_th)
        # diagonal flip
        boxes = RotatedBoxes(th_boxes)
        flipped_boxes_th = torch.Tensor([45, 10, 10, 20,
                                         np.pi / 6]).reshape(1, 1, 5)
        boxes.flip_(img_shape, direction='diagonal')
        assert_allclose(boxes.tensor, flipped_boxes_th)

    def test_translate(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        boxes = RotatedBoxes(th_boxes)
        boxes.translate_([23, 46])
        translated_boxes_th = torch.Tensor([63, 86, 10, 20,
                                            np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(boxes.tensor, translated_boxes_th)

    def test_clip(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        img_shape = [13, 14]
        boxes = RotatedBoxes(th_boxes)
        boxes.clip_(img_shape)
        cliped_boxes_th = torch.Tensor([40, 40, 10, 20,
                                        np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(boxes.tensor, cliped_boxes_th)

    def test_rotate(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        center = (50, 40)
        angle = -60
        boxes = RotatedBoxes(th_boxes)
        boxes.rotate_(center, angle)
        rotated_boxes_th = torch.Tensor(
            [45, 40 + 5 * sqrt(3), 10, 20, -np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(boxes.tensor, rotated_boxes_th)

    def test_project(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        boxes1 = RotatedBoxes(th_boxes)
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
        assert_allclose(
            boxes1.regularize_boxes('oc'), boxes2.regularize_boxes('oc'))

    def test_rescale(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        scale_factor = [0.4, 0.4]

        boxes = RotatedBoxes(th_boxes)
        boxes.rescale_(scale_factor)
        rescaled_boxes_th = torch.Tensor([16, 16, 4, 8,
                                          np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(boxes.tensor, rescaled_boxes_th)

    def test_resize(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 1, 5)
        scale_factor = [0.4, 0.8]
        boxes = RotatedBoxes(th_boxes)
        boxes.resize_(scale_factor)
        resized_boxes_th = torch.Tensor([40, 40, 4, 16,
                                         np.pi / 6]).reshape(1, 1, 5)
        assert_allclose(boxes.tensor, resized_boxes_th)

    def test_is_inside(self):
        th_boxes = torch.Tensor([[10, 10, 10, 20, 0.3], [25, 25, 10, 20, 0.2],
                                 [35, 35, 10, 20, 0.1]]).reshape(1, 3, 5)
        img_shape = [30, 30]
        boxes = RotatedBoxes(th_boxes)
        index = boxes.is_inside(img_shape)
        index_th = torch.BoolTensor([True, True, False]).reshape(1, 3)
        assert_allclose(index, index_th)

    def test_find_inside_points(self):
        th_boxes = torch.Tensor([40, 40, 10, 20, np.pi / 6]).reshape(1, 5)
        boxes = RotatedBoxes(th_boxes)
        points = torch.Tensor([[20, 40], [30, 40], [37.5, 40], [40, 40]])
        index = boxes.find_inside_points(points)
        index_th = torch.BoolTensor([False, False, True, True]).reshape(4, 1)
        assert_allclose(index, index_th)
        # test is_aligned
        boxes = boxes.expand(4, 5)
        index = boxes.find_inside_points(points, is_aligned=True)
        index_th = torch.BoolTensor([False, False, True, True])
        assert_allclose(index, index_th)

    def test_from_instance_masks(self):
        bitmap_masks = BitmapMasks.random()
        boxes = RotatedBoxes.from_instance_masks(bitmap_masks)
        self.assertIsInstance(boxes, RotatedBoxes)
        self.assertEqual(len(boxes), len(bitmap_masks))
        polygon_masks = PolygonMasks.random()
        boxes = RotatedBoxes.from_instance_masks(polygon_masks)
        self.assertIsInstance(boxes, RotatedBoxes)
        self.assertEqual(len(boxes), len(bitmap_masks))
        # zero length masks
        bitmap_masks = BitmapMasks.random(num_masks=0)
        boxes = RotatedBoxes.from_instance_masks(bitmap_masks)
        self.assertIsInstance(boxes, RotatedBoxes)
        self.assertEqual(len(boxes), 0)
        polygon_masks = PolygonMasks.random(num_masks=0)
        boxes = RotatedBoxes.from_instance_masks(polygon_masks)
        self.assertIsInstance(boxes, RotatedBoxes)
        self.assertEqual(len(boxes), 0)
