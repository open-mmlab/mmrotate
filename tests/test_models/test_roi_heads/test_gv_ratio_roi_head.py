# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.config import Config
from parameterized import parameterized

from mmrotate.registry import MODELS
from mmrotate.testing import demo_mm_inputs, demo_mm_proposals
from mmrotate.utils import register_all_modules


def _fake_roi_head():
    """Set a fake roi head config."""
    roi_head = Config(
        dict(
            type='GVRatioRoIHead',
            bbox_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='GVBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=15,
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
                    type='mmdet.SmoothL1Loss', beta=1.0 / 3.0,
                    loss_weight=1.0),
                loss_ratio=dict(
                    type='mmdet.SmoothL1Loss',
                    beta=1.0 / 3.0,
                    loss_weight=16.0)),
            train_cfg=dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='QBbox2HBboxOverlaps2D')),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                nms_pre=2000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms_rotated', iou_threshold=0.1),
                max_per_img=2000)))
    return roi_head


class TestGVRatioRoIHead(TestCase):

    def setUp(self):
        register_all_modules()

    def test_init(self):
        """Test init GV RoI head."""
        roi_head_cfg = _fake_roi_head()
        roi_head = MODELS.build(roi_head_cfg)
        self.assertTrue(roi_head.with_bbox)

    def test_gv_roi_head_loss(self):
        """Tests gv_ratio roi head loss when truth is empty and non-empty."""
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')
        s = 256
        roi_head_cfg = _fake_roi_head()
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.cuda()
        feats = []
        for i in range(len(roi_head.bbox_roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 1, s // (2**(i + 2)),
                           s // (2**(i + 2))).to(device='cuda'))
        feats = tuple(feats)

        # When truth is non-empty then both cls, and box loss
        # should be nonzero for random inputs
        image_shapes = [(3, s, s)]
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[1],
            num_classes=4,
            use_box_type=True,
            use_qbox=True,
            device='cuda')['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes,
            num_proposals=100,
            use_box_type=True,
            device='cuda')

        out = roi_head.loss(feats, proposals_list, batch_data_samples)
        loss_cls = out['loss_cls']
        loss_bbox = out['loss_bbox']
        self.assertGreater(loss_cls.sum(), 0, 'cls loss should be non-zero')
        self.assertGreater(loss_bbox.sum(), 0, 'box loss should be non-zero')

        # When there is no truth, the cls loss should be nonzero but
        # there should be no box loss.
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[0],
            num_classes=4,
            use_box_type=True,
            use_qbox=True,
            device='cuda')['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes,
            num_proposals=100,
            use_box_type=True,
            device='cuda')
        out = roi_head.loss(feats, proposals_list, batch_data_samples)
        empty_cls_loss = out['loss_cls']
        empty_bbox_loss = out['loss_bbox']
        self.assertGreater(empty_cls_loss.sum(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_bbox_loss.sum(), 0,
            'there should be no box loss when there are no true boxes')

    @parameterized.expand(['cpu', 'cuda'])
    def test_gv_ratio_roi_head_predict(self, device):
        """Tests gv_ratio roi head predict."""
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        roi_head_cfg = _fake_roi_head()
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.to(device=device)
        s = 256
        feats = []
        for i in range(len(roi_head.bbox_roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 256, s // (2**(i + 2)),
                           s // (2**(i + 2))).to(device=device))

        image_shapes = [(3, s, s)]
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[0],
            num_classes=4,
            with_mask=True,
            device=device)['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device=device)
        roi_head.predict(feats, proposals_list, batch_data_samples)

    @parameterized.expand(['cpu', 'cuda'])
    def test_gv_ratio_roi_head_forward(self, device):
        """Tests gv ratio roi head forward."""
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        roi_head_cfg = _fake_roi_head()
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.to(device=device)
        s = 256
        feats = []
        for i in range(len(roi_head.bbox_roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 256, s // (2**(i + 2)),
                           s // (2**(i + 2))).to(device=device))

        image_shapes = [(3, s, s)]
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device=device)
        roi_head.forward(feats, proposals_list)
