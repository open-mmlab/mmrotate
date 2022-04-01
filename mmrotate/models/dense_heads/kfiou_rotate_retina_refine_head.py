# Copyright (c) SJTU. All rights reserved.
import torch
from mmcv.runner import force_fp32

from ..builder import ROTATED_HEADS
from .kfiou_rotate_retina_head import KFIoURRetinaHead


@ROTATED_HEADS.register_module()
class KFIoURRetinaRefineHead(KFIoURRetinaHead):
    """Rotational Anchor-based refine head. The difference from
    `RRetinaRefineHead` is that its loss_bbox requires bbox_pred, bbox_targets,
    pred_decode and targets_decode as inputs.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='PseudoAnchorGenerator',
                     strides=[8, 16, 32, 64, 128]),
                 bbox_coder=dict(
                     type='DeltaXYWHABBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        self.bboxes_as_anchors = None
        super(KFIoURRetinaRefineHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg,
            **kwargs)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds, rois):
        """Refine predicted bounding boxes at each position of the feature
        maps. This method will be used in R3Det in refinement stages.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)
            rois (list[list[Tensor]]): input rbboxes of each level of each
             image. rois output by former stages and are to be refined

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level of each \
                image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        bboxes_list = [[] for _ in range(num_imgs)]
        assert rois is not None
        mlvl_rois = [torch.cat(r) for r in zip(*rois)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            rois = mlvl_rois[lvl]
            assert bbox_pred.size(1) == 5
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(-1, 5)
            refined_bbox = self.bbox_coder.decode(rois, bbox_pred)
            refined_bbox = refined_bbox.reshape(num_imgs, -1, 5)
            for img_id in range(num_imgs):
                bboxes_list[img_id].append(refined_bbox[img_id].detach())
        return bboxes_list

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            bboxes_as_anchors (list[list[Tensor]]) bboxes of levels of images.
                before further regression just like anchors.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple (list[Tensor]):

                - anchor_list (list[Tensor]): Anchors of each image
                - valid_flag_list (list[Tensor]): Valid flags of each image
        """
        anchor_list = [[
            bboxes_img_lvl.clone().detach() for bboxes_img_lvl in bboxes_img
        ] for bboxes_img in self.bboxes_as_anchors]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             rois=None,
             gt_bboxes_ignore=None):
        """Loss function of KFIoURRetinaRefineHead."""
        assert rois is not None
        self.bboxes_as_anchors = rois
        return super(KFIoURRetinaRefineHead, self).loss(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   rois=None):
        """Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): size / scale info for each image
            cfg (mmcv.Config): test / postprocessing configuration
            rois (list[list[Tensor]]): input rbboxes of each level of each
             image. rois output by former stages and are to be refined
            rescale (bool): if True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (xc, yc, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.
        """
        num_levels = len(cls_scores)
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        result_list = []

        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                rois[img_id], img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list
