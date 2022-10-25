# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrotate.datasets import CocoRboxDataset


class TestCocoRboxDataset(unittest.TestCase):

    def test_coco_dataset(self):
        # test CocoDataset
        metainfo = dict(CLASSES=('plane', 'bridge'), task_name='new_task')
        dataset = CocoRboxDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_rbox/coco_rbox_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        self.assertEqual(dataset.metainfo['CLASSES'], ('plane', 'bridge'))
        self.assertEqual(dataset.metainfo['task_name'], 'new_task')
        self.assertListEqual(dataset.get_cat_ids(0), [0, 1])

    def test_coco_dataset_without_filter_cfg(self):
        # test CocoRboxDataset without filter_cfg
        dataset = CocoRboxDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_rbox/coco_rbox_sample.json',
            pipeline=[])
        self.assertEqual(len(dataset), 4)

        # test with test_mode = True
        dataset = CocoRboxDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_rbox/coco_rbox_sample.json',
            test_mode=True,
            pipeline=[])
        self.assertEqual(len(dataset), 4)
