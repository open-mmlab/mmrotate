# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np
from mmdet.structures.mask import BitmapMasks

from mmrotate.datasets.transforms import ConvertMask2BoxType
from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes


class TestConvertMask2BoxType(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
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
