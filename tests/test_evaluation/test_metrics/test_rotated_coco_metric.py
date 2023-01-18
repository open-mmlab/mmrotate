# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import numpy as np
import pycocotools.mask as mask_util
import torch
from mmengine.fileio import dump

from mmrotate.evaluation import RotatedCocoMetric


class TestRotatedCocoMetric(TestCase):

    def _create_dummy_coco_json(self, json_name):
        dummy_mask = np.zeros((10, 10), order='F', dtype=np.uint8)
        dummy_mask[:5, :5] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
        image = {
            'id': 0,
            'width': 640,
            'height': 640,
            'file_name': 'fake_name.jpg',
        }

        annotation_1 = {
            'id': 1,
            'image_id': 0,
            'category_id': 0,
            'area': 400,
            'bbox': [50, 60, 20, 20, 0.],
            'iscrowd': 0,
            'segmentation': rle_mask,
        }

        annotation_2 = {
            'id': 2,
            'image_id': 0,
            'category_id': 0,
            'area': 900,
            'bbox': [100, 120, 30, 30, 0.],
            'iscrowd': 0,
            'segmentation': rle_mask,
        }

        annotation_3 = {
            'id': 3,
            'image_id': 0,
            'category_id': 1,
            'area': 1600,
            'bbox': [150, 160, 40, 40, 0.],
            'iscrowd': 0,
            'segmentation': rle_mask,
        }

        annotation_4 = {
            'id': 4,
            'image_id': 0,
            'category_id': 0,
            'area': 10000,
            'bbox': [250, 260, 100, 100, 0.],
            'iscrowd': 0,
            'segmentation': rle_mask,
        }

        categories = [
            {
                'id': 0,
                'name': 'car',
                'supercategory': 'car',
            },
            {
                'id': 1,
                'name': 'bicycle',
                'supercategory': 'bicycle',
            },
        ]

        fake_json = {
            'images': [image],
            'annotations':
            [annotation_1, annotation_2, annotation_3, annotation_4],
            'categories': categories
        }

        dump(fake_json, json_name)

    def _create_dummy_results(self):
        bboxes = np.array([[50, 60, 20, 20, 0.], [100, 120, 30, 30, 0.],
                           [150, 160, 40, 40, 0.], [250, 260, 100, 100, 0.]])
        scores = np.array([1.0, 0.98, 0.96, 0.95])
        labels = np.array([0, 0, 1, 0])
        dummy_mask = np.zeros((4, 10, 10), dtype=np.uint8)
        dummy_mask[:, :5, :5] = 1
        return dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels),
            masks=torch.from_numpy(dummy_mask))

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        with self.assertRaisesRegex(KeyError, 'metric should be one of'):
            RotatedCocoMetric(ann_file=fake_json_file, metric='unknown')

    def test_evaluate(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        # test single coco dataset evaluation
        coco_metric = RotatedCocoMetric(
            ann_file=fake_json_file,
            classwise=False,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate(size=1)
        target = {
            'r_coco/bbox_mAP': 1.0,
            'r_coco/bbox_mAP_50': 1.0,
            'r_coco/bbox_mAP_75': 1.0,
            'r_coco/bbox_mAP_s': 1.0,
            'r_coco/bbox_mAP_m': 1.0,
            'r_coco/bbox_mAP_l': 1.0,
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.bbox.json')))

        # test box and segm coco dataset evaluation
        coco_metric = RotatedCocoMetric(
            ann_file=fake_json_file,
            metric=['bbox', 'segm'],
            classwise=False,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate(size=1)
        target = {
            'r_coco/bbox_mAP': 1.0,
            'r_coco/bbox_mAP_50': 1.0,
            'r_coco/bbox_mAP_75': 1.0,
            'r_coco/bbox_mAP_s': 1.0,
            'r_coco/bbox_mAP_m': 1.0,
            'r_coco/bbox_mAP_l': 1.0,
            'r_coco/segm_mAP': 1.0,
            'r_coco/segm_mAP_50': 1.0,
            'r_coco/segm_mAP_75': 1.0,
            'r_coco/segm_mAP_s': 1.0,
            'r_coco/segm_mAP_m': 1.0,
            'r_coco/segm_mAP_l': 1.0,
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.bbox.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.segm.json')))

        # test invalid custom metric_items
        with self.assertRaisesRegex(KeyError,
                                    'metric item "invalid" is not supported'):
            coco_metric = RotatedCocoMetric(
                ann_file=fake_json_file, metric_items=['invalid'])
            coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
            coco_metric.process({}, [
                dict(
                    pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))
            ])
            coco_metric.evaluate(size=1)

        # test custom metric_items
        coco_metric = RotatedCocoMetric(
            ann_file=fake_json_file, metric_items=['mAP_m'])
        coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate(size=1)
        target = {
            'r_coco/bbox_mAP_m': 1.0,
        }
        self.assertDictEqual(eval_results, target)

    def test_classwise_evaluate(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        # test single coco dataset evaluation
        coco_metric = RotatedCocoMetric(
            ann_file=fake_json_file, metric='bbox', classwise=True)
        coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate(size=1)
        target = {
            'r_coco/bbox_mAP': 1.0,
            'r_coco/bbox_mAP_50': 1.0,
            'r_coco/bbox_mAP_75': 1.0,
            'r_coco/bbox_mAP_s': 1.0,
            'r_coco/bbox_mAP_m': 1.0,
            'r_coco/bbox_mAP_l': 1.0,
            'r_coco/car_precision': 1.0,
            'r_coco/bicycle_precision': 1.0,
        }
        self.assertDictEqual(eval_results, target)

    def test_manually_set_iou_thrs(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)

        # test single coco dataset evaluation
        coco_metric = RotatedCocoMetric(
            ann_file=fake_json_file, metric='bbox', iou_thrs=[0.3, 0.6])
        coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
        self.assertEqual(coco_metric.iou_thrs, [0.3, 0.6])

    def test_evaluate_without_json(self):
        dummy_pred = self._create_dummy_results()

        dummy_mask = np.zeros((10, 10), order='F', dtype=np.uint8)
        dummy_mask[:5, :5] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
        instances = [{
            'bbox_label': 0,
            'bbox': [40., 50., 40., 70., 60., 70., 60., 50],
            'ignore_flag': 0,
            'mask': rle_mask,
        }, {
            'bbox_label': 0,
            'bbox': [85., 105., 85., 135., 115., 135, 115., 105.],
            'ignore_flag': 0,
            'mask': rle_mask,
        }, {
            'bbox_label': 1,
            'bbox': [130., 140., 130., 180., 170., 180., 170., 140.],
            'ignore_flag': 0,
            'mask': rle_mask,
        }, {
            'bbox_label': 0,
            'bbox': [200., 210., 200., 310., 300., 310., 300., 210.],
            'ignore_flag': 0,
            'mask': rle_mask,
        }]
        coco_metric = RotatedCocoMetric(
            ann_file=None,
            metric=['bbox', 'segm'],
            classwise=False,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
        coco_metric.process({}, [
            dict(
                pred_instances=dummy_pred,
                img_id=0,
                ori_shape=(640, 640),
                instances=instances)
        ])
        eval_results = coco_metric.evaluate(size=1)
        print(eval_results)
        target = {
            'r_coco/bbox_mAP': 1.0,
            'r_coco/bbox_mAP_50': 1.0,
            'r_coco/bbox_mAP_75': 1.0,
            'r_coco/bbox_mAP_s': 1.0,
            'r_coco/bbox_mAP_m': 1.0,
            'r_coco/bbox_mAP_l': 1.0,
            'r_coco/segm_mAP': 1.0,
            'r_coco/segm_mAP_50': 1.0,
            'r_coco/segm_mAP_75': 1.0,
            'r_coco/segm_mAP_s': 1.0,
            'r_coco/segm_mAP_m': 1.0,
            'r_coco/segm_mAP_l': 1.0,
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.bbox.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.segm.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.gt.json')))

    def test_evaluate_proposal(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        coco_metric = RotatedCocoMetric(
            ann_file=fake_json_file, metric='proposal')
        coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate(size=1)
        target = {
            'r_coco/AR@100': 1,
            'r_coco/AR@300': 1.0,
            'r_coco/AR@1000': 1.0,
            'r_coco/AR_s@1000': 1.0,
            'r_coco/AR_m@1000': 1.0,
            'r_coco/AR_l@1000': 1.0
        }
        self.assertDictEqual(eval_results, target)

    def test_format_only(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        with self.assertRaises(AssertionError):
            RotatedCocoMetric(
                ann_file=fake_json_file,
                classwise=False,
                format_only=True,
                outfile_prefix=None)

        coco_metric = RotatedCocoMetric(
            ann_file=fake_json_file,
            metric='bbox',
            classwise=False,
            format_only=True,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate(size=1)
        self.assertDictEqual(eval_results, dict())
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/test.bbox.json'))
