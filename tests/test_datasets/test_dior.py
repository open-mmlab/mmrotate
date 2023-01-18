# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrotate.datasets import DIORDataset


class TestDIORDataset(unittest.TestCase):

    def test_dior(self):
        dataset = DIORDataset(
            data_root='tests/data/dior/',
            ann_file='demo.txt',
            data_prefix=dict(img_path='JPEGImages/'),
            filter_cfg=dict(
                filter_empty_gt=True, min_size=32, bbox_min_size=4),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)

        data_list = dataset.load_data_list()
        self.assertEqual(len(data_list), 1)
        self.assertEqual(data_list[0]['img_id'], '00001')
        self.assertEqual(data_list[0]['img_path'].replace('\\', '/'),
                         'tests/data/dior/JPEGImages/00001.jpg')
        self.assertEqual(
            data_list[0]['xml_path'].replace('\\', '/'),
            'tests/data/dior/Annotations/Oriented Bounding Boxes/00001.xml')
        self.assertEqual(len(data_list[0]['instances']), 1)
        self.assertEqual(dataset.get_cat_ids(0), [9])
        self.assertEqual(len(dataset._metainfo['classes']), 20)
