# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmrotate.models.roi_heads.bbox_heads import GVBBoxHead
from mmrotate.utils import register_all_modules


class TestGVBBoxHead(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand(['cpu', 'cuda'])
    def test_forward_loss(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        bbox_head = GVBBoxHead(
            in_channels=1,
            fc_out_channels=1,
            roi_feat_size=7,
            num_classes=4,
            ratio_thr=0.8,
            bbox_coder=dict(
                type='DeltaXYWHQBBoxCoder',
                target_means=(.0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2)),
            fix_coder=dict(type='GVFixCoder'),
            ratio_coder=dict(type='GVRatioCoder'),
            predict_box_type='rbox',
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_fix=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 3.0, loss_weight=1.0),
            loss_ratio=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 3.0, loss_weight=16.0))
        bbox_head = bbox_head.to(device=device)

        num_samples = 4
        feats = torch.rand((num_samples, 1, 7, 7)).to(device)
        bbox_head(x=feats)
