# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmdet.models import L1Loss
from mmengine.structures import InstanceData
from parameterized import parameterized

from mmrotate.models.dense_heads import (RotatedRTMDetHead,
                                         RotatedRTMDetSepBNHead)
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules


class TestRotatedRTMDetHead(unittest.TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand([(RotatedRTMDetHead, ), (RotatedRTMDetSepBNHead, )])
    def test_rotated_rtmdet_head_loss(self, head_cls):
        """Tests rotated rtmdet head loss when truth is empty and non-empty."""
        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')

        angle_version = 'le90'
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        rtm_head = head_cls(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            angle_version=angle_version,
            anchor_generator=dict(
                type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16,
                                                                    32]),
            bbox_coder=dict(
                type='DistanceAnglePointCoder', angle_version=angle_version),
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(
                type='RotatedIoULoss', mode='linear', loss_weight=2.0),
            with_objectness=False,
            pred_kernel_size=1,
            use_hbbox_loss=False,
            scale_angle=False,
            loss_angle=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'),
            train_cfg=dict(
                assigner=dict(
                    type='mmdet.DynamicSoftLabelAssigner',
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    topk=13),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)).cuda()

        # Rotated RTMDet head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0]).cuda()
            for stride in rtm_head.prior_generator.strides)
        cls_scores, bbox_preds, angle_preds = rtm_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 5)).cuda()
        gt_instances.labels = torch.LongTensor([]).cuda()

        empty_gt_losses = rtm_head.loss_by_feat(cls_scores, bbox_preds,
                                                angle_preds, [gt_instances],
                                                img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(
            torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]).cuda())
        gt_instances.labels = torch.LongTensor([2]).cuda()

        one_gt_losses = rtm_head.loss_by_feat(cls_scores, bbox_preds,
                                              angle_preds, [gt_instances],
                                              img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')

        # Test head with angle_loss
        rtm_head.loss_angle = L1Loss(loss_weight=0.2)
        with_ang_losses = rtm_head.loss_by_feat(cls_scores, bbox_preds,
                                                angle_preds, [gt_instances],
                                                img_metas)
        with_ang_cls_loss = sum(with_ang_losses['loss_cls'])
        with_ang_box_loss = sum(with_ang_losses['loss_bbox'])
        with_ang_ang_loss = sum(with_ang_losses['loss_angle'])

        self.assertGreater(with_ang_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(with_ang_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(with_ang_ang_loss, 0,
                           'angle loss should be non-zero')

    @parameterized.expand([(RotatedRTMDetHead, ), (RotatedRTMDetSepBNHead, )])
    def test_rotated_rtmdet_head_loss_with_hbb(self, head_cls):
        """Tests rotated rtmdet head loss when truth is empty and non-empty."""
        angle_version = 'le90'
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        rtm_head = head_cls(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            angle_version=angle_version,
            anchor_generator=dict(
                type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16,
                                                                    32]),
            bbox_coder=dict(
                type='DistanceAnglePointCoder', angle_version=angle_version),
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
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
            with_objectness=False,
            pred_kernel_size=1,
            use_hbbox_loss=True,
            scale_angle=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'),
            train_cfg=dict(
                assigner=dict(
                    type='mmdet.DynamicSoftLabelAssigner',
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    topk=13),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))

        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in rtm_head.prior_generator.strides)
        cls_scores, bbox_preds, angle_preds = rtm_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 5))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = rtm_head.loss_by_feat(cls_scores, bbox_preds,
                                                angle_preds, [gt_instances],
                                                img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        empty_ang_loss = sum(empty_gt_losses['loss_angle'])
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_ang_loss, 0,
            'there should be no angle loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(
            torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]))
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = rtm_head.loss_by_feat(cls_scores, bbox_preds,
                                              angle_preds, [gt_instances],
                                              img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        onegt_ang_loss = sum(one_gt_losses['loss_angle'])
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_ang_loss, 0, 'angle loss should be non-zero')
