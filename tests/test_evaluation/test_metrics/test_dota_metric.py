# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmrotate.evaluation import DOTAMetric


class TestDOTAMetric(unittest.TestCase):

    def _create_dummy_results(self):
        bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                           [100, 120, 10.0, 20.0, 0.1], 
                           [150, 160, 10.0, 20.0, 0.2],
                           [250, 260, 10.0, 20.0, 0.3]])
        scores = np.array([1.0, 0.98, 0.96, 0.95])
        labels = np.array([0, 0, 0, 0])
        return dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels))

    def _create_dummy_data_sample(self):
        bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                           [100, 120, 10.0, 20.0, 0.1], 
                           [150, 160, 10.0, 20.0, 0.2],
                           [250, 260, 10.0, 20.0, 0.3]])
        labels = np.array([0] * 4)
        bboxes_ignore = np.array([[0] * 5])
        labels_ignore = np.array([0])
        return dict(gt_instances=dict(
                        bboxes=torch.from_numpy(bboxes),
                        labels=torch.from_numpy(labels)),
                    ignored_instances=dict(
                        bboxes=torch.from_numpy(bboxes_ignore),
                        labels=torch.from_numpy(labels_ignore)))

    def test_init(self):
        # test invalid iou_thrs
        with self.assertRaises(AssertionError):
            DOTAMetric(iou_thrs={'a', 0.5})

        metric = DOTAMetric(iou_thrs=0.6)
        self.assertEqual(metric.iou_thrs, [0.6])

    def test_eval(self):
        metric = DOTAMetric()
        metric.dataset_meta = {'CLASSES': ('plane',)}
        metric.process(
            data_batch=[dict(inputs=None, data_sample=self._create_dummy_data_sample())],
            predictions=[dict(pred_instances=self._create_dummy_results())])
        results = metric.evaluate(size=1)
        targets = {'dota/AP50': 1.0, 'dota/mAP': 1.0}
        self.assertDictEqual(results, targets)

        # test multi-threshold
        metric = DOTAMetric(iou_thrs=[0.1, 0.5])
        metric.dataset_meta = {'CLASSES': ('plane',)}
        metric.process(
            data_batch=[dict(inputs=None, data_sample=self._create_dummy_data_sample())],
            predictions=[dict(pred_instances=self._create_dummy_results())])
        results = metric.evaluate(size=1)
        targets = {
            'dota/AP10': 1.0,
            'dota/AP50': 1.0,
            'dota/mAP': 1.0
        }
        self.assertDictEqual(results, targets)
