# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmrotate.models.task_modules.assigners import RotatedATSSAssigner
from mmrotate.structures.bbox import RotatedBoxes


class TestATSSAssigner(TestCase):

    def test_atss_assigner(self):
        atss_assigner = RotatedATSSAssigner(
            topk=9, iou_calculator=dict(type='RBboxOverlaps2D'))
        priors = torch.FloatTensor([
            [5, 5, 5, 5, 0.1],
            [15, 15, 5, 5, 0.],
            [10, 10, 5, 5, 0.1],
            [35, 35, 3, 3, 0.],
        ])
        gt_bboxes = torch.FloatTensor([
            [5, 4.5, 5, 4.5, 0.],
            [5, 15, 5, 4, 0.],
        ])
        gt_labels = torch.LongTensor([2, 3])
        priors = RotatedBoxes(priors, clone=False)
        gt_bboxes = RotatedBoxes(gt_bboxes, clone=False)

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_bboxes = [4]

        assign_result = atss_assigner.assign(pred_instances, num_level_bboxes,
                                             gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 4)
        self.assertEqual(len(assign_result.labels), 4)

        expected_gt_inds = torch.LongTensor([1, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_atss_assigner_with_ignore(self):
        atss_assigner = RotatedATSSAssigner(
            topk=9, iou_calculator=dict(type='RBboxOverlaps2D'))
        priors = torch.FloatTensor([
            [5, 5, 5, 5, 0.1],
            [15, 15, 5, 5, 0.],
            [10, 10, 5, 5, 0.1],
            [35, 35, 3, 3, 0.],
        ])
        gt_bboxes = torch.FloatTensor([
            [5, 4.5, 5, 4.5, 0.],
            [5, 15, 5, 4, 0.],
        ])
        gt_labels = torch.LongTensor([2, 3])
        gt_bboxes_ignore = torch.Tensor([
            [35, 35, 5, 5, 0.],
        ])
        priors = RotatedBoxes(priors, clone=False)
        gt_bboxes = RotatedBoxes(gt_bboxes, clone=False)
        gt_bboxes_ignore = RotatedBoxes(gt_bboxes_ignore, clone=False)

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)
        num_level_bboxes = [4]
        assign_result = atss_assigner.assign(
            pred_instances,
            num_level_bboxes,
            gt_instances,
            gt_instances_ignore=gt_instances_ignore)

        expected_gt_inds = torch.LongTensor([1, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_atss_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        atss_assigner = RotatedATSSAssigner(
            topk=9, iou_calculator=dict(type='RBboxOverlaps2D'))
        priors = torch.FloatTensor([
            [5, 5, 5, 5, 0.1],
            [15, 15, 5, 5, 0.],
            [10, 10, 5, 5, 0.1],
            [35, 35, 3, 3, 0.],
        ])
        gt_bboxes = torch.empty(0, 5)
        gt_labels = torch.empty(0)
        priors = RotatedBoxes(priors, clone=False)
        gt_bboxes = RotatedBoxes(gt_bboxes, clone=False)

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_bboxes = [4]
        assign_result = atss_assigner.assign(pred_instances, num_level_bboxes,
                                             gt_instances)

        expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_atss_assigner_with_empty_boxes(self):
        """Test corner case where a network might predict no boxes."""
        atss_assigner = RotatedATSSAssigner(
            topk=9, iou_calculator=dict(type='RBboxOverlaps2D'))
        priors = torch.empty((0, 5))
        gt_bboxes = torch.FloatTensor([
            [5, 4.5, 5, 4.5, 0.],
            [5, 15, 5, 4, 0.],
        ])
        gt_labels = torch.LongTensor([2, 3])
        priors = RotatedBoxes(priors, clone=False)
        gt_bboxes = RotatedBoxes(gt_bboxes, clone=False)

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_bboxes = [0]
        assign_result = atss_assigner.assign(pred_instances, num_level_bboxes,
                                             gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 0)
        self.assertTrue(tuple(assign_result.labels.shape) == (0, ))

    def test_atss_assigner_with_empty_boxes_and_ignore(self):
        """Test corner case where a network might predict no boxes and
        ignore_iof_thr is on."""
        atss_assigner = RotatedATSSAssigner(
            topk=9, iou_calculator=dict(type='RBboxOverlaps2D'))
        priors = torch.empty((0, 4))
        gt_bboxes = torch.FloatTensor([
            [5, 4.5, 5, 4.5, 0.],
            [5, 15, 5, 4, 0.],
        ])
        gt_bboxes_ignore = torch.Tensor([
            [35, 35, 5, 5, 0.],
        ])
        gt_labels = torch.LongTensor([2, 3])
        priors = RotatedBoxes(priors, clone=False)
        gt_bboxes = RotatedBoxes(gt_bboxes, clone=False)

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)
        num_level_bboxes = [0]

        assign_result = atss_assigner.assign(
            pred_instances,
            num_level_bboxes,
            gt_instances,
            gt_instances_ignore=gt_instances_ignore)
        self.assertEqual(len(assign_result.gt_inds), 0)
        self.assertTrue(tuple(assign_result.labels.shape) == (0, ))

    def test_atss_assigner_with_empty_boxes_and_gt(self):
        """Test corner case where a network might predict no boxes and no
        gt."""
        atss_assigner = RotatedATSSAssigner(
            topk=9, iou_calculator=dict(type='RBboxOverlaps2D'))
        priors = torch.empty((0, 5))
        gt_bboxes = torch.empty((0, 5))
        gt_labels = torch.empty(0)
        priors = RotatedBoxes(priors, clone=False)
        gt_bboxes = RotatedBoxes(gt_bboxes, clone=False)
        num_level_bboxes = [0]

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        assign_result = atss_assigner.assign(pred_instances, num_level_bboxes,
                                             gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 0)
