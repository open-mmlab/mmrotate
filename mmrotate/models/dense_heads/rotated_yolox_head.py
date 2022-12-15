# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import bbox_cxcywh_to_xyxy, multi_apply, reduce_mean
from mmdet.models.dense_heads import YOLOXHead

from mmrotate.core import build_bbox_coder, norm_angle
from ..builder import ROTATED_HEADS, build_loss


@ROTATED_HEADS.register_module()
class RotatedYOLOXHead(YOLOXHead):
    """Rotated YOLOXHead head used in `YOLOX.

    <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        separate_angle (bool): If true, angle prediction is separated from
            bbox regression loss. Default: False.
        with_angle_l1 (bool): If true, compute L1 loss with angle.
            Default: True.
        angle_norm_factor (float): Regularization factor of angle. Only
            used when with_angle_l1 is True
        angle_coder (dict): Config of angle coder.
        loss_angle (dict): Config of angle loss, only used when
            separate_angle is True.
    """

    def __init__(self,
                 separate_angle=False,
                 with_angle_l1=True,
                 angle_norm_factor=3.14,
                 edge_swap=None,
                 angle_coder=dict(type='PseudoAngleCoder'),
                 loss_angle=None,
                 **kwargs):

        self.angle_coder = build_bbox_coder(angle_coder)
        self.angle_len = self.angle_coder.coding_len

        super().__init__(**kwargs)

        self.separate_angle = separate_angle
        self.with_angle_l1 = with_angle_l1
        self.angle_norm_factor = angle_norm_factor
        self.edge_swap = edge_swap
        if self.edge_swap:
            assert self.edge_swap in ['oc', 'le90', 'le135']
        if self.separate_angle:
            assert loss_angle is not None, \
                'loss_angle must be specified when separate_angle is True'
            self.loss_angle = build_loss(loss_angle)

    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        self.multi_level_conv_ang = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj, conv_ang = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)
            self.multi_level_conv_ang.append(conv_ang)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls, conv_reg, conv_obj = super(RotatedYOLOXHead,
                                             self)._build_predictor()
        conv_ang = nn.Conv2d(self.feat_channels, self.angle_len, 1)
        return conv_cls, conv_reg, conv_obj, conv_ang

    def init_weights(self):
        super(RotatedYOLOXHead, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_ang in self.multi_level_conv_ang:
            conv_ang.bias.data.fill_(bias_init)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg,
                       conv_obj, conv_ang):
        """Forward feature of a single scale level."""

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        angle_pred = conv_ang(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, angle_pred, objectness,

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        return multi_apply(
            self.forward_single, feats, self.multi_level_cls_convs,
            self.multi_level_reg_convs, self.multi_level_conv_cls,
            self.multi_level_conv_reg, self.multi_level_conv_obj,
            self.multi_level_conv_ang)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'objectnesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angles for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, a) and the 5-th column
                is a score between 0 and 1. The second item is a (n,) tensor
                 where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array(
            [img_meta['scale_factor'] for img_meta in img_metas])

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                   self.angle_len)
            for angle_pred in angle_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_decoded_angle = self.angle_coder.decode(
            flatten_angle_preds).unsqueeze(-1)

        flatten_hbboxes_cxcywh, flatten_decoded_angle = \
            self._bbox_decode_cxcywha(
                flatten_priors, flatten_bbox_preds, flatten_decoded_angle)

        flatten_rbboxes = torch.cat(
            [flatten_hbboxes_cxcywh, flatten_decoded_angle], dim=-1)

        if rescale:
            flatten_rbboxes[..., :4] /= flatten_rbboxes.new_tensor(
                scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_rbboxes[img_id]

            result_list.append(
                self._bboxes_nms(cls_scores, bboxes, score_factor, cfg))

        return result_list

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    def _bbox_decode_cxcywha(self, priors, bbox_preds, decoded_angle):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        if self.edge_swap:
            w = whs[..., 0:1]
            h = whs[..., 1:2]
            w_regular = torch.where(w > h, w, h)
            h_regular = torch.where(w > h, h, w)
            theta_regular = torch.where(w > h, decoded_angle,
                                        decoded_angle + np.pi / 2)
            theta_regular = norm_angle(theta_regular, self.edge_swap)
            return torch.cat([xys, w_regular, h_regular],
                             dim=-1), theta_regular
        else:
            return torch.cat([xys, whs], -1), decoded_angle

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            angle_preds (list[Tensor]): Box angles for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [x, y, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                   self.angle_len)
            for angle_pred in angle_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)

        flatten_decoded_angle = self.angle_coder.decode(
            flatten_angle_preds).unsqueeze(-1)

        flatten_hbboxes_cxcywh, flatten_decoded_angle = \
            self._bbox_decode_cxcywha(
                flatten_priors, flatten_bbox_preds, flatten_decoded_angle)

        flatten_rbboxes = torch.cat(
            [flatten_hbboxes_cxcywh, flatten_decoded_angle], dim=-1)
        flatten_hbboxes = bbox_cxcywh_to_xyxy(flatten_hbboxes_cxcywh)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_rbboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        # Loss Bbox
        if self.separate_angle:
            hbbox_xyxy_targets = bbox_cxcywh_to_xyxy(bbox_targets[..., :4])
            angle_targets = bbox_targets[..., 4:5]
            angle_targets = self.angle_coder.encode(angle_targets)

            loss_bbox = self.loss_bbox(
                flatten_hbboxes.view(-1, 4)[pos_masks],
                hbbox_xyxy_targets) / num_total_samples
            loss_angle = self.loss_angle(
                flatten_angle_preds.view(-1, self.angle_len)[pos_masks],
                angle_targets) / num_total_samples
        else:
            loss_bbox = self.loss_bbox(
                flatten_rbboxes.view(-1, 5)[pos_masks],
                bbox_targets) / num_total_samples

        # Loss Objectness and Loss Cls
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets) / num_total_samples

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.separate_angle:
            loss_dict.update(loss_angle=loss_angle)

        # Loss L1
        if self.use_l1:
            if self.with_angle_l1:
                flatten_rbbox_preds = torch.cat([
                    flatten_bbox_preds,
                    flatten_decoded_angle / self.angle_norm_factor
                ],
                                                dim=-1)
                loss_l1 = self.loss_l1(
                    flatten_rbbox_preds.view(-1, 5)[pos_masks],
                    l1_targets) / num_total_samples
            else:
                loss_l1 = self.loss_l1(
                    flatten_bbox_preds.view(-1, 4)[pos_masks],
                    l1_targets) / num_total_samples

            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 5] in [x, y, w, h, a]
                format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 5] in [x, y, w, h, a] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 5))
            if self.with_angle_l1:
                l1_target = cls_preds.new_zeros((0, 5))
            else:
                l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        return super(RotatedYOLOXHead,
                     self)._get_target_single(cls_preds, objectness, priors,
                                              decoded_bboxes, gt_bboxes,
                                              gt_labels)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = gt_bboxes[..., :4]
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        if self.with_angle_l1:
            angle_target = gt_bboxes[..., 4:5] / self.angle_norm_factor
            return torch.cat([l1_target, angle_target], dim=-1)
        else:
            return l1_target
