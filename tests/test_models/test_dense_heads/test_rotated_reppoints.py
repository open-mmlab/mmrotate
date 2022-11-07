# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmdet.structures import DetDataSample
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads import RotatedRepPointsHead
from mmrotate.utils import register_all_modules


class TestRotatedRepPointsHead(unittest.TestCase):

    def setUp(self):
        register_all_modules()

    def test_head_loss(self):
        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')

        cfg = ConfigDict(
            dict(
                num_classes=2,
                in_channels=32,
                point_feat_channels=10,
                num_points=9,
                gradient_mul=0.3,
                point_strides=[8, 16, 32, 64, 128],
                point_base_scale=2,
                loss_cls=dict(
                    type='mmdet.FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.375),
                loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
                transform_method='rotrect'),
            train_cfg=dict(
                init=dict(
                    assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False),
                refine=dict(
                    assigner=dict(
                        type='MaxConvexIoUAssigner',
                        pos_iou_thr=0.4,
                        neg_iou_thr=0.3,
                        min_pos_iou=0,
                        ignore_iof_thr=-1),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False)),
            test_cfg=dict(
                nms_pre=2000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms_rotated', iou_threshold=0.4),
                max_per_img=2000))
        reppoints_head = RotatedRepPointsHead(**cfg).cuda()
        s = 256
        img_metas = [{
            'img_shape': (s, s),
            'scale_factor': (1, 1),
            'pad_shape': (s, s),
            'batch_input_shape': (s, s)
        }]
        x = [
            torch.rand(1, 32, s // 2**(i + 2), s // 2**(i + 2)).cuda()
            for i in range(5)
        ]

        # Don't support empty ground truth now.

        reppoints_head.train()
        forward_outputs = reppoints_head.forward(x)

        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor([[
            100.6326, 70.8874, 130.6667, 70.8874, 130.6667, 86.8757, 100.6326,
            86.8757
        ]]).cuda()
        gt_instances.labels = torch.LongTensor([2]).cuda()
        gt_bboxes_ignore = None
        one_gt_losses = reppoints_head.loss_by_feat(*forward_outputs,
                                                    [gt_instances], img_metas,
                                                    gt_bboxes_ignore)
        # loss_cls should all be non-zero
        self.assertTrue(
            all([loss.item() > 0 for loss in one_gt_losses['loss_cls']]))
        # only one level loss_pts_init is non-zero
        cnt_non_zero = 0
        for loss in one_gt_losses['loss_pts_init']:
            if loss.item() != 0:
                cnt_non_zero += 1
        self.assertEqual(cnt_non_zero, 1)

        # only one level loss_pts_refine is non-zero
        cnt_non_zero = 0
        for loss in one_gt_losses['loss_pts_init']:
            if loss.item() != 0:
                cnt_non_zero += 1
        self.assertEqual(cnt_non_zero, 1)

        # test loss
        samples = DetDataSample()
        samples.set_metainfo(img_metas[0])
        samples.gt_instances = gt_instances
        reppoints_head.loss(x, [samples])
        # test only predict
        reppoints_head.eval()
        reppoints_head.predict(x, [samples], rescale=True)
