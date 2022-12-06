# Copyright (c) OpenMMLab. All rights reserved.
import math
import unittest

import pytest
import torch
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads import H2RBoxHead
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules


class TestH2RBoxHead(unittest.TestCase):

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
        h2rbox_head = H2RBoxHead(
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
            loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
            loss_angle=None,
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            square_classes=[9, 11],
            crop_size=(1024, 1024),
            rotation_agnostic_classes=None,
            weak_supervised=True,
            loss_bbox_ss=dict(
                type='H2RBoxConsistencyLoss',
                loss_weight=0.4,
                center_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=0.0),
                shape_loss_cfg=dict(type='mmdet.IoULoss', loss_weight=1.0),
                angle_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=1.0)),
            norm_cfg=None).cuda()

        # Fcos head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0]).cuda()
            for stride in h2rbox_head.prior_generator.strides)
        cls_scores, bbox_preds, angle_preds, centernesses = \
            h2rbox_head.forward(feats)
        rot = (torch.rand(1, device=angle_preds[0].device) * 2 - 1) * math.pi

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 5)).cuda()
        gt_instances.labels = torch.LongTensor([]).cuda()

        empty_gt_losses = h2rbox_head.loss_by_feat(cls_scores, bbox_preds,
                                                   angle_preds, centernesses,
                                                   bbox_preds, angle_preds,
                                                   rot, [gt_instances],
                                                   img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss, centerness loss and consistency loss should be zero
        empty_cls_loss = empty_gt_losses['loss_cls'].item()
        empty_box_loss = empty_gt_losses['loss_bbox'].item()
        empty_ctr_loss = empty_gt_losses['loss_centerness'].item()
        empty_box_ss_loss = empty_gt_losses['loss_bbox_ss'].item()
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_ctr_loss, 0,
            'there should be no centerness loss when there are no true boxes')
        self.assertEqual(
            empty_box_ss_loss, 0,
            'there should be no consistency loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss, centerness loss
        # and consistency loss should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = RotatedBoxes(
            torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0]]).cuda())
        gt_instances.labels = torch.LongTensor([2]).cuda()

        one_gt_losses = h2rbox_head.loss_by_feat(cls_scores, bbox_preds,
                                                 angle_preds, centernesses,
                                                 bbox_preds, angle_preds, rot,
                                                 [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].item()
        onegt_box_loss = one_gt_losses['loss_bbox'].item()
        onegt_ctr_loss = one_gt_losses['loss_centerness'].item()
        onegt_box_ss_loss = one_gt_losses['loss_bbox_ss'].item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(onegt_box_ss_loss, 0,
                           'consistency loss should be non-zero')

        # # Test the `center_sampling` works fine.
        h2rbox_head.center_sampling = True
        ctrsamp_losses = h2rbox_head.loss_by_feat(cls_scores, bbox_preds,
                                                  angle_preds, centernesses,
                                                  bbox_preds, angle_preds, rot,
                                                  [gt_instances], img_metas)
        ctrsamp_cls_loss = ctrsamp_losses['loss_cls'].item()
        ctrsamp_box_loss = ctrsamp_losses['loss_bbox'].item()
        ctrsamp_ctr_loss = ctrsamp_losses['loss_centerness'].item()
        ctrsamp_box_ss_loss = ctrsamp_losses['loss_bbox_ss'].item()
        self.assertGreater(ctrsamp_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(ctrsamp_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(ctrsamp_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(ctrsamp_box_ss_loss, 0,
                           'consistency loss should be non-zero')

        # Test the `norm_on_bbox` works fine.
        h2rbox_head.norm_on_bbox = True
        normbox_losses = h2rbox_head.loss_by_feat(cls_scores, bbox_preds,
                                                  angle_preds, centernesses,
                                                  bbox_preds, angle_preds, rot,
                                                  [gt_instances], img_metas)
        normbox_cls_loss = normbox_losses['loss_cls'].item()
        normbox_box_loss = normbox_losses['loss_bbox'].item()
        normbox_ctr_loss = normbox_losses['loss_centerness'].item()
        normbox_box_ss_loss = normbox_losses['loss_bbox_ss'].item()
        self.assertGreater(normbox_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(normbox_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(normbox_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(normbox_box_ss_loss, 0,
                           'consistency loss should be non-zero')

        # Test the `rotation_agnostic_classes` works fine.
        h2rbox_head.rotation_agnostic_classes = [9, 11]
        rac_losses = h2rbox_head.loss_by_feat(cls_scores, bbox_preds,
                                              angle_preds, centernesses,
                                              bbox_preds, angle_preds, rot,
                                              [gt_instances], img_metas)
        rac_cls_loss = rac_losses['loss_cls'].item()
        rac_box_loss = rac_losses['loss_bbox'].item()
        rac_ctr_loss = rac_losses['loss_centerness'].item()
        rac_box_ss_loss = rac_losses['loss_bbox_ss'].item()
        self.assertGreater(rac_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(rac_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(rac_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(rac_box_ss_loss, 0,
                           'consistency loss should be non-zero')

        # # Test head without weak_supervised
        h2rbox_head = H2RBoxHead(
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
            square_classes=[9, 11],
            crop_size=(1024, 1024),
            rotation_agnostic_classes=None,
            weak_supervised=False,
            loss_bbox_ss=dict(
                type='H2RBoxConsistencyLoss',
                loss_weight=0.4,
                center_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=0.0),
                shape_loss_cfg=dict(type='mmdet.IoULoss', loss_weight=1.0),
                angle_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=1.0)),
            norm_cfg=None).cuda()
        # h2rbox_head.weak_supervised = False
        # h2rbox_head.loss_bbox = dict(type='RotatedIoULoss', loss_weight=1.0)
        without_ws_losses = h2rbox_head.loss_by_feat(cls_scores, bbox_preds,
                                                     angle_preds, centernesses,
                                                     bbox_preds, angle_preds,
                                                     rot, [gt_instances],
                                                     img_metas)
        without_ws_cls_loss = without_ws_losses['loss_cls'].item()
        without_ws_box_loss = without_ws_losses['loss_bbox'].item()
        without_ws_ctr_loss = without_ws_losses['loss_centerness'].item()
        without_ws_box_ss_loss = without_ws_losses['loss_bbox_ss'].item()

        self.assertGreater(without_ws_cls_loss, 0,
                           'cls loss should be non-zero')
        self.assertGreater(without_ws_box_loss, 0,
                           'box loss should be non-zero')
        self.assertGreater(without_ws_ctr_loss, 0,
                           'centerness loss should be non-zero')
        self.assertGreater(without_ws_box_ss_loss, 0,
                           'consistency loss should be non-zero')
