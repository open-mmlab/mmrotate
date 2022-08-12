# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrotate.datasets import SARDataset


class TestDOTADataset(unittest.TestCase):

    def test_sar(self):
        dataset = SARDataset(
            metainfo=dict(CLASSES=('plane', )),
            data_root='tests/data/sar/',
            ann_file='./',
            data_prefix=dict(img_path='images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)

        data_list = dataset.load_data_list()
        self.assertEqual(len(data_list), 1)
        self.assertEqual(data_list[0]['img_id'], 'demo')
        self.assertEqual(data_list[0]['file_name'], 'demo.png')
        self.assertEqual(data_list[0]['img_path'],
                         'tests/data/sar/images/demo.png')
        self.assertEqual(data_list[0]['width'], 400)
        self.assertEqual(data_list[0]['height'], 300)
        self.assertEqual(len(data_list[0]['instances']), 4)
        self.assertEqual(dataset.get_cat_ids(0), [0, 0, 0, 0])
