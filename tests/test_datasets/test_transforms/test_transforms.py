# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmengine.testing import assert_allclose

from mmrotate.datasets.transforms import (ConvertBoxType, ConvertMask2BoxType,
                                          RandomChoiceRotate, RandomRotate,
                                          Rotate)
from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes


def construct_toy_data(poly2mask, use_box_type=False):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    results['img'] = img
    results['img_shape'] = img.shape[:2]
    if use_box_type:
        results['gt_bboxes'] = RotatedBoxes(
            np.array([[1, 0, 2, 3, 0.]], dtype=np.float32))
    else:
        results['gt_bboxes'] = np.array([[1, 0, 2, 2, 0.]], dtype=np.float32)
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

    def setUp(self):
        self.data_info = dict(
            gt_bboxes=QuadriBoxes(
                torch.Tensor([[10, 10, 20, 10, 20, 20, 10, 20]])))

    def test_convert(self):
        transform = ConvertBoxType(box_type_mapping=dict(gt_bboxes='rbox'))
        results = transform(copy.deepcopy(self.data_info))
        self.assertIsInstance(results['gt_bboxes'], RotatedBoxes)


class TestConvertMask2BoxType(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.data_info = dict(
            ori_shape=(1333, 800),
            gt_bboxes=np.array([[0, 0, 112, 112]], dtype=np.float32),
            gt_masks=BitmapMasks(
                rng.rand(1, 1333, 800), height=1333, width=800),
            instances=[{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'mask': [[0, 0, 0, 20, 10, 20, 10, 0]],
                'ignore_flag': 0
            }])

    def test_convert(self):
        # test keep_mask is True
        transform = ConvertMask2BoxType('rbox', keep_mask=True)
        results = transform(copy.deepcopy(self.data_info))
        self.assertIsInstance(results['gt_bboxes'], RotatedBoxes)
        self.assertEqual(len(results['instances'][0]['bbox']), 5)
        self.assertEqual(results['gt_masks'].masks.shape[1:], (1333, 800))

        # test keep_mask is False
        transform = ConvertMask2BoxType('rbox')
        results = transform(copy.deepcopy(self.data_info))
        self.assertIsInstance(results['gt_bboxes'], RotatedBoxes)
        self.assertIsNone(results.get('gt_masks', None))
        self.assertIsNone(results['instances'][0].get('mask', None))

        # test convert to qbox
        transform = ConvertMask2BoxType('qbox')
        results = transform(copy.deepcopy(self.data_info))
        self.assertIsInstance(results['gt_bboxes'], QuadriBoxes)
        self.assertEqual(len(results['instances'][0]['bbox']), 8)
        self.assertIsNone(results['instances'][0].get('mask', None))


class TestRotate(unittest.TestCase):

    def setUp(self):
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask_boxtype = construct_toy_data(
            poly2mask=True, use_box_type=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_rotate_use_box_type(self):
        # test case when no rotate aug (rotate_angle=0)
        transform = Rotate(
            rotate_angle=0,
            img_border_value=self.img_border_value,
            seg_ignore_label=self.seg_ignore_label,
        )
        results_wo_rotate = transform(copy.deepcopy(self.results_mask_boxtype))
        check_result_same(self.results_mask_boxtype, results_wo_rotate,
                          self.check_keys)

        # test clockwise rotation with angle 90
        transform = Rotate(rotate_angle=90)
        results_rotated = transform(copy.deepcopy(self.results_mask_boxtype))
        # The image, masks, and semantic segmentation map
        # will be bilinearly interpolated.
        img_gt = np.array([[5, 8, 4, 1], [5, 9, 5, 1], [6, 10, 6,
                                                        2]]).astype(np.uint8)
        img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
        results_gt = copy.deepcopy(self.results_mask_boxtype)
        results_gt['img'] = img_gt
        results_gt['gt_bboxes'] = RotatedBoxes(
            np.array([[3.5, 0.5, 2., 3., 1.5708]], dtype=np.float32))
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results_gt['gt_masks'] = BitmapMasks(gt_masks, 3, 4)
        results_gt['gt_ignore_flags'] = np.array(np.array([1], dtype=bool))
        results_gt['gt_seg_map'] = np.array(
            [[255, 13, 13, 13], [255, 255, 13, 255],
             [255, 255, 255,
              255]]).astype(self.results_mask_boxtype['gt_seg_map'].dtype)
        check_result_same(results_gt, results_rotated, self.check_keys)

    def test_repr(self):
        transform = Rotate(rotate_angle=180)
        self.assertEqual(
            repr(transform), ('Rotate(rotate_angle=180, '
                              'img_border_value=(0.0, 0.0, 0.0), '
                              'mask_border_value=0, '
                              'seg_ignore_label=255, '
                              'interpolation=bilinear)'))


class TestRandomRotate(unittest.TestCase):

    def setUp(self):
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask_boxtype = construct_toy_data(
            poly2mask=True, use_box_type=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_random_rotate(self):
        # test case when no rotate aug (rotate_angle=0)
        transform = RandomRotate(prob=0., rotate_angle=90)
        results_wo_rotate = transform(copy.deepcopy(self.results_mask_boxtype))
        check_result_same(self.results_mask_boxtype, results_wo_rotate,
                          self.check_keys)

        transform = RandomRotate(prob=1., rotate_angle=90)
        results_rotated = transform(copy.deepcopy(self.results_mask_boxtype))
        # bboxes may be clipped when rotating
        self.assertIn(results_rotated['gt_bboxes'].size(0), [0, 1])

        transform = RandomRotate(
            prob=1., rotate_angle=90, rect_obj_labels=[13])
        results_rotated = transform(copy.deepcopy(self.results_mask_boxtype))
        self.assertIn(transform.rotate.rotate_angle, [90, 180, -90, -180])

    def test_repr(self):
        transform = RandomRotate(rotate_angle=180)
        self.assertEqual(
            repr(transform),
            ('RandomRotate(prob=0.5, '
             'rotate_angle=180, '
             'rect_obj_labels=None, '
             "rotate_cfg={'type': 'Rotate', 'rotate_angle': 180})"))


class TestRandomChoiceRotate(unittest.TestCase):

    def setUp(self):
        self.check_keys = ('img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                           'gt_ignore_flags', 'gt_seg_map')
        self.results_mask_boxtype = construct_toy_data(
            poly2mask=True, use_box_type=True)
        self.img_border_value = (104, 116, 124)
        self.seg_ignore_label = 255

    def test_random_choice_rotate(self):
        # test case when no rotate aug (rotate_angle=0)
        transform = RandomChoiceRotate(
            angles=[30, 60],
            prob=0.,
        )
        results_wo_rotate = transform(copy.deepcopy(self.results_mask_boxtype))
        check_result_same(self.results_mask_boxtype, results_wo_rotate,
                          self.check_keys)

        transform = RandomChoiceRotate(angles=[30, 60], prob=1.)
        transform(copy.deepcopy(self.results_mask_boxtype))
        self.assertIn(transform.rotate.rotate_angle, [30, 60])

        transform = RandomChoiceRotate(
            angles=[30, 60], prob=1., rect_obj_labels=[13])
        transform(copy.deepcopy(self.results_mask_boxtype))
        self.assertIn(transform.rotate.rotate_angle, [90, 180, -90, -180])

    def test_repr(self):
        transform = RandomChoiceRotate(angles=[180])
        self.assertEqual(
            repr(transform), ('RandomChoiceRotate(angles=[180], '
                              'prob=0.5, '
                              'rect_obj_labels=None, '
                              "rotate_cfg={'type': 'Rotate'})"))
