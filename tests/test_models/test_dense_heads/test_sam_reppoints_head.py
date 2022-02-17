# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import pytest
import torch

from mmrotate.models.dense_heads import SAMRepPointsHead


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('reassign', [True, False])
def test_sam_head_loss(reassign):
    """Tests sam head loss when truth is empty and non-empty.

    Args:
        reassign (bool): If True, reassign samples.
    """
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            init=dict(
                assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            refine=dict(
                assigner=dict(type='SASAssigner', topk=3),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)))
    self = SAMRepPointsHead(
        num_classes=15,
        in_channels=1,
        feat_channels=1,
        point_feat_channels=1,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='BCConvexGIoULoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
        transform_method='rotrect',
        topk=6,
        anti_factor=0.75,
        train_cfg=train_cfg).cuda()
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size).cuda()
        for feat_size in [4, 8, 16, 32, 64]
    ]
    cls_scores, pts_out_init, pts_out_refine = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 5)).cuda()]
    gt_labels = [torch.LongTensor([]).cuda()]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, pts_out_init, pts_out_refine,
                                gt_bboxes, gt_labels, img_metas,
                                gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_pts_init_loss = sum(empty_gt_losses['loss_pts_init'])
    empty_pts_refine_loss = sum(empty_gt_losses['loss_pts_refine'])
    assert empty_cls_loss.item() != 0, 'cls loss should be non-zero'
    assert empty_pts_init_loss.item() == 0, (
        'there should be no pts_init loss when there are no true boxes')
    assert empty_pts_refine_loss.item() == 0, (
        'there should be no pts_refine loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874, 0.]]).cuda(),
    ]
    gt_labels = [torch.LongTensor([2]).cuda()]
    one_gt_losses = self.loss(cls_scores, pts_out_init, pts_out_refine,
                              gt_bboxes, gt_labels, img_metas,
                              gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_pts_init_loss = sum(one_gt_losses['loss_pts_init'])
    onegt_pts_refine_loss = sum(one_gt_losses['loss_pts_refine'])
    assert onegt_cls_loss.item() != 0, 'cls loss should be non-zero'
    assert onegt_pts_init_loss.item() >= 0, 'pts_init loss should be non-zero'
    assert onegt_pts_refine_loss.item() >= 0, (
        'pts_refine loss should be non-zero')
