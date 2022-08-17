# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np

from mmrotate.datasets.pipelines import LoadPatchfromNDArray


class TestLoadPatchFromNDArray(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = {'img': np.zeros((256, 256, 3), dtype=np.uint8)}

    def test_transform(self):
        transform = LoadPatchfromNDArray()
        results = copy.deepcopy(self.results)
        results['patch'] = [64, 64, 128, 128]
        results = transform(results)
        self.assertEqual(results['img'].shape, (64, 64, 3))
        self.assertEqual(results['img'].dtype, np.uint8)
        self.assertEqual(results['img_shape'], (64, 64))
        self.assertEqual(results['ori_shape'], (64, 64))

        transform = LoadPatchfromNDArray(pad_val=(1, 2, 3))
        results = copy.deepcopy(self.results)
        results['patch'] = [-10, -10, 266, 266]
        results = transform(results)
        self.assertEqual(results['img'].shape, (276, 276, 3))
        self.assertEqual(results['img'].dtype, np.uint8)
        self.assertEqual(results['img_shape'], (276, 276))
        self.assertEqual(results['ori_shape'], (276, 276))
        self.assertEqual(tuple(results['img'][0, 0].tolist()), (1, 2, 3))

    def test_repr(self):
        transform = LoadPatchfromNDArray()
        repr(transform)
