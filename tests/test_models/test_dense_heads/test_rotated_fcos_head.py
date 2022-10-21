# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import pytest
import torch
from mmdet.models import L1Loss
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads import RotatedFCOSHead
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules


class TestFCOSHead(unittest.TestCase):

    def setUp(self):
        register_all_modules()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    def test_rotated_fcos_head_loss(self):
        """Tests fcos head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        fcos_head = RotatedFCOSHead(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            use_hbbox_loss=False,
            scale_angle=True,
            bbox_coder=dict(
                type='DistanceAnglePointCoder', angle_version='le90'),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
            loss_angle=None,
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            norm_cfg=None).cuda()

        # Fcos head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0]).cuda()
            for stride in fcos_head.prior_generator.strides)
        cls_scores, bbox_preds, angle_preds, centernesses = fcos_head.forward(
            feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 5)).cuda()
        gt_instances.labels = torch.LongTensor([]).cuda()

        empty_gt_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                 angle_preds, centernesses,
                                                 [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_loss = empty_gt_losses['loss_cls'].item()
        empty_box_loss = empty_gt_losses['loss_bbox'].item()
        empty_ctr_loss = empty_gt_losses['loss_centerness'].item()
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_ctr_loss, 0,
            'there should be no centerness loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(
            torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]).cuda())
        gt_instances.labels = torch.LongTensor([2]).cuda()

        one_gt_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                               angle_preds, centernesses,
                                               [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].item()
        onegt_box_loss = one_gt_losses['loss_bbox'].item()
        onegt_ctr_loss = one_gt_losses['loss_centerness'].item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_ctr_loss, 0,
                           'centerness loss should be non-zero')

        # Test the `center_sampling` works fine.
        fcos_head.center_sampling = True
        ctrsamp_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                angle_preds, centernesses,
                                                [gt_instances], img_metas)
        ctrsamp_cls_loss = ctrsamp_losses['loss_cls'].item()
        ctrsamp_box_loss = ctrsamp_losses['loss_bbox'].item()
        ctrsamp_ctr_loss = ctrsamp_losses['loss_centerness'].item()
        self.assertGreater(ctrsamp_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(ctrsamp_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(ctrsamp_ctr_loss, 0,
                           'centerness loss should be non-zero')

        # Test the `norm_on_bbox` works fine.
        fcos_head.norm_on_bbox = True
        normbox_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                angle_preds, centernesses,
                                                [gt_instances], img_metas)
        normbox_cls_loss = normbox_losses['loss_cls'].item()
        normbox_box_loss = normbox_losses['loss_bbox'].item()
        normbox_ctr_loss = normbox_losses['loss_centerness'].item()
        self.assertGreater(normbox_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(normbox_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(normbox_ctr_loss, 0,
                           'centerness loss should be non-zero')

        # Test head with angle_loss
        fcos_head.loss_angle = L1Loss(loss_weight=0.2)
        with_ang_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                 angle_preds, centernesses,
                                                 [gt_instances], img_metas)
        with_ang_cls_loss = with_ang_losses['loss_cls'].item()
        with_ang_box_loss = with_ang_losses['loss_bbox'].item()
        with_ang_ctr_loss = with_ang_losses['loss_centerness'].item()
        with_ang_ang_loss = with_ang_losses['loss_angle'].item()

        self.assertGreater(with_ang_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(with_ang_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(with_ang_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(with_ang_ang_loss, 0,
                           'angle loss should be non-zero')

    def test_rotated_fcos_head_loss_with_hbb(self):
        """Tests fcos head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        fcos_head = RotatedFCOSHead(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            use_hbbox_loss=True,
            scale_angle=False,
            bbox_coder=dict(
                type='DistanceAnglePointCoder', angle_version='le90'),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
            angle_coder=dict(
                type='CSLCoder',
                angle_version='le90',
                omega=1,
                window='gaussian',
                radius=1),
            loss_angle=dict(
                type='SmoothFocalLoss', gamma=2.0, alpha=0.25,
                loss_weight=0.2),
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            norm_cfg=None)

        # Fcos head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in fcos_head.prior_generator.strides)
        cls_scores, bbox_preds, angle_preds, centernesses = fcos_head.forward(
            feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 5))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                 angle_preds, centernesses,
                                                 [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_loss = empty_gt_losses['loss_cls'].item()
        empty_box_loss = empty_gt_losses['loss_bbox'].item()
        empty_ctr_loss = empty_gt_losses['loss_centerness'].item()
        empty_ang_loss = empty_gt_losses['loss_angle'].item()
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_ctr_loss, 0,
            'there should be no centerness loss when there are no true boxes')
        self.assertEqual(
            empty_ang_loss, 0,
            'there should be no angle loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(
            torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]))
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                               angle_preds, centernesses,
                                               [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].item()
        onegt_box_loss = one_gt_losses['loss_bbox'].item()
        onegt_ctr_loss = one_gt_losses['loss_centerness'].item()
        onegt_ang_loss = one_gt_losses['loss_angle'].item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(onegt_ang_loss, 0, 'angle loss should be non-zero')

        # Test the `center_sampling` works fine.
        fcos_head.center_sampling = True
        ctrsamp_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                angle_preds, centernesses,
                                                [gt_instances], img_metas)
        ctrsamp_cls_loss = ctrsamp_losses['loss_cls'].item()
        ctrsamp_box_loss = ctrsamp_losses['loss_bbox'].item()
        ctrsamp_ctr_loss = ctrsamp_losses['loss_centerness'].item()
        ctrsamp_ang_loss = ctrsamp_losses['loss_angle'].item()
        self.assertGreater(ctrsamp_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(ctrsamp_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(ctrsamp_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(ctrsamp_ang_loss, 0,
                           'angle loss should be non-zero')

        # Test the `norm_on_bbox` works fine.
        fcos_head.norm_on_bbox = True
        normbox_losses = fcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                angle_preds, centernesses,
                                                [gt_instances], img_metas)
        normbox_cls_loss = normbox_losses['loss_cls'].item()
        normbox_box_loss = normbox_losses['loss_bbox'].item()
        normbox_ctr_loss = normbox_losses['loss_centerness'].item()
        normbox_ang_loss = normbox_losses['loss_angle'].item()
        self.assertGreater(normbox_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(normbox_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(normbox_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(normbox_ang_loss, 0,
                           'angle loss should be non-zero')
