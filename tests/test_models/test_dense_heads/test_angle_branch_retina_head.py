# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads import AngleBranchRetinaHead
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules


class TestAngleBranchRetinaHead(TestCase):

    def setUp(self):
        register_all_modules()

    def test_angle_branch_retina_head_loss(self):
        """Tests anchor head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        cfg = Config(
            dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(type='mmdet.PseudoSampler'
                             ),  # Focal loss should use PseudoSampler
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        bbox_head = AngleBranchRetinaHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            anchor_generator=dict(
                type='FakeRotatedAnchorGenerator',
                angle_version='le90',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[1.0, 0.5, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version='le90',
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                use_box_type=False),
            angle_coder=dict(
                type='CSLCoder',
                angle_version='le90',
                omega=4,
                window='gaussian',
                radius=3),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0),
            loss_angle=dict(
                type='SmoothFocalLoss', gamma=2.0, alpha=0.25,
                loss_weight=0.8),
            train_cfg=cfg)

        # Anchor head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
            for i in range(len(bbox_head.prior_generator.strides)))
        cls_scores, bbox_preds, angle_preds = bbox_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(torch.empty((0, 5)))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = bbox_head.loss_by_feat(cls_scores, bbox_preds,
                                                 angle_preds, [gt_instances],
                                                 img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # there should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        empty_angle_loss = sum(empty_gt_losses['loss_angle'])
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_angle_loss.item(), 0,
            'there should be no angle loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(
            torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]))
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = bbox_head.loss_by_feat(cls_scores, bbox_preds,
                                               angle_preds, [gt_instances],
                                               img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        onegt_angle_loss = sum(one_gt_losses['loss_angle'])
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_angle_loss.item(), 0,
                           'angle loss should be non-zero')
