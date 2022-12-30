# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrotate.datasets import HRSCDataset


class TestHRSCDataset(unittest.TestCase):

    def test_hrsc(self):
        dataset = HRSCDataset(
            data_root='tests/data/hrsc/',
            ann_file='demo.txt',
            data_prefix=dict(sub_data_root='FullDataSet/'),
            filter_cfg=dict(
                filter_empty_gt=True, min_size=32, bbox_min_size=4),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)

        data_list = dataset.load_data_list()
        self.assertEqual(len(data_list), 1)
        self.assertEqual(data_list[0]['img_id'], '100000006')
        self.assertEqual(
            data_list[0]['img_path'].replace('\\', '/'),
            'tests/data/hrsc/FullDataSet/AllImages/100000006.bmp')
        self.assertEqual(
            data_list[0]['xml_path'].replace('\\', '/'),
            'tests/data/hrsc/FullDataSet/Annotations/100000006.xml')
        self.assertEqual(len(data_list[0]['instances']), 1)
        self.assertEqual(dataset.get_cat_ids(0), [0])
        self.assertEqual(dataset._metainfo['classes'], ('ship', ))

    def test_hrsc_classwise(self):
        dataset = HRSCDataset(
            data_root='tests/data/hrsc/',
            ann_file='demo.txt',
            data_prefix=dict(sub_data_root='FullDataSet/'),
            classwise=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)

        data_list = dataset.load_data_list()
        self.assertEqual(len(data_list), 1)
        self.assertEqual(data_list[0]['img_id'], '100000006')
        self.assertEqual(
            data_list[0]['img_path'].replace('\\', '/'),
            'tests/data/hrsc/FullDataSet/AllImages/100000006.bmp')
        self.assertEqual(
            data_list[0]['xml_path'].replace('\\', '/'),
            'tests/data/hrsc/FullDataSet/Annotations/100000006.xml')
        self.assertEqual(len(data_list[0]['instances']), 1)
        self.assertEqual(dataset.get_cat_ids(0), [12])
        self.assertEqual(len(dataset._metainfo['classes']), 31)
