# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.structures import InstanceData
from parameterized import parameterized

from mmrotate.models.task_modules.assigners import MaxConvexIoUAssigner


class TestMaxConvexIoUAssigner(unittest.TestCase):

    @parameterized.expand([(0.5, ), ((0, 0.5), )])
    def test_max_convex_iou_assigner(self, neg_iou_thr):

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')

        assigner = MaxConvexIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=neg_iou_thr,
        )
        priors = torch.FloatTensor([
            [0, 0, 2, 2, 5, 5, 10, 10, 10, 5, 10, 0, 8, 0, 5, 0, 2, 0],
            [10, 0, 12, 2, 15, 5, 20, 10, 20, 5, 20, 0, 18, 0, 15, 0, 12, 0],
            [
                10, 10, 12, 12, 15, 15, 20, 20, 20, 15, 20, 10, 18, 10, 15, 10,
                12, 10
            ],
            [
                12, 10, 14, 12, 17, 15, 22, 20, 22, 15, 22, 10, 20, 10, 17, 10,
                14, 10
            ],
        ]).cuda()
        gt_bboxes = torch.FloatTensor([
            [5, 5, 5, 4, 0.1],
            [5, 15, 5, 5, 0.0],
        ]).cuda()
        gt_labels = torch.LongTensor([2, 3]).cuda()

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)

        assign_result = assigner.assign(pred_instances, gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 4)
        self.assertEqual(len(assign_result.labels), 4)
