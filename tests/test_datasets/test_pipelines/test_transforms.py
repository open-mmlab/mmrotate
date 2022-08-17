# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmengine.testing import assert_allclose

from mmrotate.core.bbox.structures import QuadriBoxes, RotatedBoxes
from mmrotate.datasets.pipelines import (ConvertBoxType, RandomChoiceRotate,
                                         RandomRotate, Rotate)


def construct_toy_data(poly2mask=True):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    results['img'] = img
    results['img_shape'] = img.shape[:2]
    results['gt_bboxes'] = RotatedBoxes([[1.5, 1, 1, 2, 0]],
                                        dtype=torch.float32)
    results['gt_bboxes_labels'] = np.array([13], dtype=np.int64)
    if poly2mask:
        gt_masks = np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
    else:
        raw_masks = [[np.array([1, 2, 1, 0, 2, 1], dtype=np.float32)]]
        results['gt_masks'] = PolygonMasks(raw_masks, 3, 4)
    results['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
    results['gt_seg_map'] = np.array(
        [[255, 13, 255, 255], [255, 13, 13, 255], [255, 13, 255, 255]],
        dtype=np.uint8)
    return results


def check_result_same(results, pipeline_results, check_keys):
    """Check whether the ``pipeline_results`` is the same with the predefined
    ``results``.

    Args:
        results (dict): Predefined results which should be the standard
            output of the transform pipeline.
        pipeline_results (dict): Results processed by the transform
            pipeline.
        check_keys (tuple): Keys that need to be checked between
            results and pipeline_results.
    """
    for key in check_keys:
        if results.get(key, None) is None:
            continue
        if isinstance(results[key], (BitmapMasks, PolygonMasks)):
            assert_allclose(pipeline_results[key].to_ndarray(),
                            results[key].to_ndarray())
        elif isinstance(results[key], BaseBoxes):
            assert_allclose(pipeline_results[key].tensor, results[key].tensor)
        else:
            assert_allclose(pipeline_results[key], results[key])


class TestConvertBoxType(unittest.TestCase):

    def test_transform(self):
        transform = ConvertBoxType({'gt_bboxes': 'qbox'})
        # test empyt input
        results = {}
        results = transform(results)
        # test convert gt_bboxes
        th_bboxes = np.random.random((4, 5))
        results = dict(gt_bboxes=RotatedBoxes(th_bboxes))
        results = transform(results)
        self.assertIsInstance(results['gt_bboxes'], QuadriBoxes)

    def test_repr(self):
        transform = ConvertBoxType({'gt_bboxes': 'qbox'})
        repr(transform)


class TestRotate(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask = construct_toy_data(poly2mask=True)
        self.results_poly = construct_toy_data(poly2mask=False)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_transform(self):
        transform = Rotate(
            rotate_angle=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label)
        results_wo_rotate = transform(copy.deepcopy(self.results_mask))
        check_result_same(self.results_mask, results_wo_rotate,
                          self.check_keys)

        # test clockwise rotation with angle 90
        transform = Rotate(rotate_angle=90, img_border_value=128)
        results_rotated = transform(copy.deepcopy(self.results_mask))
        # The image, masks, and semantic segmentation map
        # will be bilinearly interpolated.
        img_gt = np.array([[69, 8, 4, 65], [69, 9, 5, 65],
                           [70, 10, 6, 66]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt = copy.deepcopy(self.results_mask)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = RotatedBoxes(
            np.array([[2.5, 1, 1, 2, np.pi / 2]], dtype=np.float32))
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 13, 13], [255, 255, 13, 255],
             [255, 255, 255,
              255]]).astype(self.results_mask['gt_seg_map'].dtype)
        check_result_same(results_gt, results_rotated, self.check_keys)

        # test clockwise rotation with angle 90, PolygonMasks
        results_rotated = transform(copy.deepcopy(self.results_poly))
        gt_masks = [[np.array([0, 1, 0, 1, 0, 2], dtype=np.float)]]
        results_gt['gt_masks'] = PolygonMasks(gt_masks, 3, 4)
        check_result_same(results_gt, results_rotated, self.check_keys)

    def test_repr(self):
        transform = Rotate(rotate_angle=90)
        repr(transform)


class TestRandomRotate(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results = construct_toy_data()

    def test_transform(self):
        with self.assertRaises(AssertionError):
            RandomRotate(angle_range=-4)
        with self.assertRaises(AssertionError):
            RandomRotate(angle_range=190)
        # no rotation
        transform = RandomRotate(prob=0)
        results = transform(copy.deepcopy(self.results))
        check_result_same(results, self.results, self.check_keys)
        # random rotation
        transform = RandomRotate(prob=1, rect_obj_labels=[13])
        results = transform(copy.deepcopy(self.results))
        self.assertIn(transform.rotate.rotate_angle,
                      transform.horizontal_angles)
        self.assertLessEqual(
            abs(transform.rotate.rotate_angle), transform.angle_range)

    def test_repr(self):
        transform = Rotate(rotate_angle=90)
        repr(transform)


class TestRandomChoiceRotate(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results = construct_toy_data()

    def test_transform(self):
        with self.assertRaises(AssertionError):
            RandomChoiceRotate([30], prob=1.4)
        with self.assertRaises(AssertionError):
            RandomChoiceRotate([30], prob=[0.7, 0.8])
        with self.assertRaises(AssertionError):
            RandomChoiceRotate([0], prob=[0.7, 0.8])
        # no rotation
        transform = RandomChoiceRotate([30], prob=0)
        results = transform(copy.deepcopy(self.results))
        check_result_same(results, self.results, self.check_keys)
        # random rotation
        transform = RandomChoiceRotate([30], prob=1)
        rotate = Rotate(rotate_angle=30)
        results = transform(copy.deepcopy(self.results))
        angle30_results = rotate(copy.deepcopy(self.results))
        check_result_same(results, angle30_results, self.check_keys)
        # List probabity
        transform = RandomChoiceRotate([30, 60], prob=[1, 0])
        rotate = Rotate(rotate_angle=30)
        results = transform(copy.deepcopy(self.results))
        angle30_results = rotate(copy.deepcopy(self.results))
        check_result_same(results, angle30_results, self.check_keys)
        # Rect_obj_labels
        transform = RandomChoiceRotate([30], prob=1, rect_obj_labels=[13])
        results = transform(copy.deepcopy(self.results))
        self.assertIn(transform.rotate.rotate_angle,
                      transform.horizontal_angles)

    def test_repr(self):
        transform = Rotate(rotate_angle=90)
        repr(transform)
