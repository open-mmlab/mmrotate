# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrotate.datasets import DOTADataset


class TestDOTADataset(unittest.TestCase):

    def test_voc2007_init(self):
        dataset = DOTADataset(
            data_root='tests/data/',
            ann_file='labelTxt/',
            data_prefix=dict(img_path='images/'),
            filter_cfg=dict(
                filter_empty_gt=True, min_size=32, bbox_min_size=32),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)

        data_list = dataset.load_data_list()
        self.assertEqual(len(data_list), 1)
        self.assertEqual(len(data_list[0]['instances']), 4)
        self.assertEqual(dataset.get_cat_ids(0), [0, 0, 0, 0])
