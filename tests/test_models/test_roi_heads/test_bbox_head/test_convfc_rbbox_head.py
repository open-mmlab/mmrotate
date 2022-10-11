# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmrotate.models.roi_heads.bbox_heads import RotatedShared2FCBBoxHead
from mmrotate.utils import register_all_modules


class TestRotatedShared2FCBBoxHead(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand(['cpu', 'cuda'])
    def test_forward_loss(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        bbox_head = RotatedShared2FCBBoxHead(
            predict_box_type='rbox',
            in_channels=1,
            fc_out_channels=1,
            roi_feat_size=7,
            num_classes=4,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version='oc',
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
                use_box_type=False),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))
        bbox_head = bbox_head.to(device=device)

        num_samples = 4
        feats = torch.rand((num_samples, 1, 7, 7)).to(device)
        bbox_head(x=feats)

    @parameterized.expand(['cpu', 'cuda'])
    def test_forward_kfloss(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        bbox_head = RotatedShared2FCBBoxHead(
            predict_box_type='rbox',
            in_channels=1,
            fc_out_channels=1,
            roi_feat_size=7,
            num_classes=4,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version='oc',
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
                use_box_type=False),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox_type='kfiou',
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0))
        bbox_head = bbox_head.to(device=device)

        num_samples = 4
        feats = torch.rand((num_samples, 1, 7, 7)).to(device)
        bbox_head(x=feats)
