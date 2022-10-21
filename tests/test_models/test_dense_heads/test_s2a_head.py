# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads import S2AHead, S2ARefineHead
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules


class TestS2AHead(TestCase):

    def setUp(self):
        register_all_modules()

    def test_s2a_head_loss(self):
        """Tests anchor head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        init_head = S2AHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            anchor_generator=dict(
                type='FakeRotatedAnchorGenerator',
                angle_version='oc',
                scales=[4],
                ratios=[1.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version='oc',
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                use_box_type=False),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0))
        cfg = Config(
            dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False))
        refine_head = S2ARefineHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            frm_cfg=dict(
                type='AlignConv',
                feat_channels=256,
                kernel_size=3,
                strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version='oc',
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0),
            train_cfg=cfg)

        # Anchor head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
            for i in range(len(init_head.prior_generator.strides)))

        # test filter_bboxes of S2AHead
        cls_scores, bbox_preds = init_head(feats)
        init_rois = init_head.filter_bboxes(cls_scores, bbox_preds)
        self.assertEqual(len(init_rois), 1)
        self.assertEqual(len(init_rois[0]), 5)
        self.assertEqual(init_rois[0][0].shape, (4096, 5))
        self.assertEqual(init_rois[0][1].shape, (1024, 5))
        self.assertEqual(init_rois[0][2].shape, (256, 5))
        self.assertEqual(init_rois[0][3].shape, (64, 5))
        self.assertEqual(init_rois[0][4].shape, (16, 5))

        # test loss_by_feat of S2ARefineHead
        feats = (
            torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
            for i in range(len(refine_head.prior_generator.strides)))
        cls_scores, bbox_preds = refine_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(torch.empty((0, 5)))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = refine_head.loss_by_feat(
            cls_scores, bbox_preds, [gt_instances], img_metas, rois=init_rois)
        # When there is no truth, the cls loss should be nonzero but
        # there should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(
            torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]))
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = refine_head.loss_by_feat(
            cls_scores, bbox_preds, [gt_instances], img_metas, rois=init_rois)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
