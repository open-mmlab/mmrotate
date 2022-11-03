# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads import RotatedATSSHead
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules


class TestRotatedATSSHead(unittest.TestCase):

    def setUp(self):
        register_all_modules()

    def test_atss_head_loss(self):
        """Tests atss head loss when truth is empty and non-empty."""
        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')

        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1
        }]
        cfg = Config(
            dict(
                assigner=dict(
                    type='RotatedATSSAssigner',
                    topk=9,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        atss_head = RotatedATSSHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            feat_channels=1,
            norm_cfg=None,
            train_cfg=cfg,
            anchor_generator=dict(
                type='FakeRotatedAnchorGenerator',
                angle_version='le90',
                octave_base_scale=4,
                scales_per_octave=1,
                ratios=[1.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version='le90',
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='RotatedIoULoss', mode='linear', loss_weight=2.0),
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)).cuda()
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size).cuda()
            for feat_size in [8, 16, 32, 64, 128]
        ]
        cls_scores, bbox_preds, centernesses = atss_head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 5)).cuda()
        gt_instances.labels = torch.LongTensor([]).cuda()

        empty_gt_losses = atss_head.loss_by_feat(cls_scores, bbox_preds,
                                                 centernesses, [gt_instances],
                                                 img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        empty_centerness_loss = sum(empty_gt_losses['loss_centerness'])
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_centerness_loss.item(), 0,
            'there should be no centerness loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(
            torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]).cuda())
        gt_instances.labels = torch.LongTensor([2]).cuda()
        one_gt_losses = atss_head.loss_by_feat(cls_scores, bbox_preds,
                                               centernesses, [gt_instances],
                                               img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        onegt_centerness_loss = sum(one_gt_losses['loss_centerness'])
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_centerness_loss.item(), 0,
                           'centerness loss should be non-zero')
