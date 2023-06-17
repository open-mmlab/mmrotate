# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.cnn import Scale
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl)
from mmdet.structures.bbox import cat_boxes, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.models.dense_heads.rotated_fcos_head import RotatedFCOSHead
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes, hbox2rbox, rbox2hbox

INF = 1e8


@MODELS.register_module()
class H2RBoxHead(RotatedFCOSHead):
    """Anchor-free head used in `H2RBox <https://arxiv.org/abs/2210.06742>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        angle_version (str): Angle representations. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Defaults to False.
        scale_angle (bool): If true, add scale to angle pred branch.
            Defaults to True.
        angle_coder (:obj:`ConfigDict` or dict): Config of angle coder.
        h_bbox_coder (dict): Config of horzional bbox coder,
            only used when use_hbbox_loss is True.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder. Defaults
            to 'DistanceAnglePointCoder'.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness loss.
        loss_angle (:obj:`ConfigDict` or dict, Optional): Config of angle loss.
        loss_bbox_ss (:obj:`ConfigDict` or dict): Config of consistency loss.
        rotation_agnostic_classes (list): Ids of rotation agnostic category.
        weak_supervised (bool): If true, horizontal gtbox is input.
            Defaults to True.
        square_classes (list): Ids of the square category.
        crop_size (tuple[int]): Crop size from image center.
            Defaults to (768, 768).

    Example:
        >>> self = H2RBoxHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, angle_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 angle_version: str = 'le90',
                 use_hbbox_loss: bool = False,
                 scale_angle: bool = True,
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 h_bbox_coder: ConfigType = dict(
                     type='mmdet.DistancePointBBoxCoder'),
                 bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='RotatedIoULoss', loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_angle: OptConfigType = None,
                 loss_bbox_ss: ConfigType = dict(
                     type='mmdet.IoULoss', loss_weight=1.0),
                 rotation_agnostic_classes: list = None,
                 weak_supervised: bool = True,
                 square_classes: list = None,
                 crop_size: Tuple[int, int] = (768, 768),
                 **kwargs):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            angle_version=angle_version,
            use_hbbox_loss=use_hbbox_loss,
            scale_angle=scale_angle,
            angle_coder=angle_coder,
            h_bbox_coder=h_bbox_coder,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_angle=loss_angle,
            **kwargs)

        self.loss_bbox_ss = MODELS.build(loss_bbox_ss)
        self.rotation_agnostic_classes = rotation_agnostic_classes
        self.weak_supervised = weak_supervised
        self.square_classes = square_classes
        self.crop_size = crop_size

    def obb2xyxy(self, rbboxes):
        w = rbboxes[:, 2::5]
        h = rbboxes[:, 3::5]
        a = rbboxes[:, 4::5]
        cosa = torch.cos(a).abs()
        sina = torch.sin(a).abs()
        hbbox_w = cosa * w + sina * h
        hbbox_h = sina * w + cosa * h
        dx = rbboxes[..., 0]
        dy = rbboxes[..., 1]
        dw = hbbox_w.reshape(-1)
        dh = hbbox_h.reshape(-1)
        x1 = dx - dw / 2
        y1 = dy - dh / 2
        x2 = dx + dw / 2
        y2 = dy + dh / 2
        return torch.stack((x1, y1, x2, y2), -1)

    def _process_rotation_agnostic(self, tensor, cls, dim=4):
        _rot_agnostic_mask = torch.ones_like(tensor)
        for c in self.rotation_agnostic_classes:
            if dim is None:
                _rot_agnostic_mask[cls == c] = 0
            else:
                _rot_agnostic_mask[cls == c, dim] = 0
        return tensor * _rot_agnostic_mask

    def forward_ss_single(self, feats: Tensor, scale: Scale,
                          stride: int) -> Tuple[Tensor, Tensor]:
        """Forward features of a single scale level in SS branch.

        Args:
            feats (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: bbox predictions and angle predictions of input
                feature maps.
        """
        reg_feat = feats
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        return bbox_pred, angle_pred

    def forward_ss(self,
                   feats: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: A tuple of each level outputs.

            - bbox_pred (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - angle_pred (list[Tensor]): Box angle for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_ss_single, feats, self.scales,
                           self.strides)

    def loss(self, x_ws: Tuple[Tensor], x_ss: Tuple[Tensor], rot: float,
             batch_gt_instances: InstanceData,
             batch_gt_instances_ignore: InstanceData,
             batch_img_metas: List[dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x_ws (tuple[Tensor]): Features from the weakly supervised network,
                each is a 4D-tensor.
            x_ss (tuple[Tensor]): Features from the self-supervised network,
                each is a 4D-tensor.
            rot (float): Angle of view rotation.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_gt_instances_ignore (list[:obj:`batch_gt_instances_ignore`]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
        Returns:
            dict: A dictionary of loss components.
        """

        cls_scores_ws, bbox_preds_ws, angle_preds_ws, centernesses_ws = self(
            x_ws)
        bbox_preds_ss, angle_preds_ss = self.forward_ss(x_ss)

        losses = self.loss_by_feat(cls_scores_ws, bbox_preds_ws,
                                   angle_preds_ws, centernesses_ws,
                                   bbox_preds_ss, angle_preds_ss, rot,
                                   batch_gt_instances, batch_img_metas,
                                   batch_gt_instances_ignore)
        return losses

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        centernesses: List[Tensor],
        bbox_preds_ss: List[Tensor],
        angle_preds_ss: List[Tensor],
        rot: float,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level in
                weakly supervised barch, each is a 4D-tensor, the channel
                number is num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level in weakly supervised barch, each is a 4D-tensor, the
                channel number is num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level in
                weakly supervised barch, each is a 4D-tensor, the channel
                number is num_points * encode_size.
            centernesses (list[Tensor]): centerness for each scale level in
                weakly supervised barch, each is a 4D-tensor, the channel
                number is num_points * 1.
            bbox_preds_ss (list[Tensor]): Box energies / deltas for each scale
                level in self-supervised barch, each is a 4D-tensor, the
                channel number is num_points * 4.
            angle_preds_ss (list[Tensor]): Box angle for each scale level in
                self-supervised barch, each is a 4D-tensor, the channel number
                is num_points * encode_size.
            rot (float): Angle of view rotation.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        assert len(bbox_preds_ss) == len(angle_preds_ss)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # bbox_targets here is in format t,b,l,r
        # angle_targets is not coded here
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        angle_dim = self.angle_coder.encode_size
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, angle_dim)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:

            cosa, sina = math.cos(rot), math.sin(rot)
            tf = flatten_cls_scores.new_tensor([[cosa, -sina], [sina, cosa]])

            pos_inds_ss = []
            pos_inds_ss_b = []
            pos_inds_ss_v = torch.empty_like(pos_inds, dtype=torch.bool)
            offset = 0
            for h, w in featmap_sizes:
                level_mask = (offset <= pos_inds).logical_and(
                    pos_inds < offset + num_imgs * h * w)
                pos_ind = pos_inds[level_mask] - offset
                xy = torch.stack((pos_ind % w, (pos_ind // w) % h), dim=-1)
                b = pos_ind // (w * h)
                ctr = tf.new_tensor([[(w - 1) / 2, (h - 1) / 2]])
                xy_ss = ((xy - ctr).matmul(tf.T) + ctr).round().long()
                x_ss = xy_ss[..., 0]
                y_ss = xy_ss[..., 1]
                xy_valid_ss = ((x_ss >= 0) & (x_ss < w) & (y_ss >= 0) &
                               (y_ss < h))
                pos_ind_ss = (b * h + y_ss) * w + x_ss
                pos_inds_ss_v[level_mask] = xy_valid_ss
                pos_inds_ss.append(pos_ind_ss[xy_valid_ss] + offset)
                pos_inds_ss_b.append(b[xy_valid_ss])
                offset += num_imgs * h * w

            has_valid_ss = pos_inds_ss_v.any()

            pos_points = flatten_points[pos_inds]
            pos_labels = flatten_labels[pos_inds]

            if has_valid_ss:
                pos_inds_ss = torch.cat(pos_inds_ss)
                # pos_inds_ss_b = torch.cat(pos_inds_ss_b)
                flatten_bbox_preds_ss = [
                    bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                    for bbox_pred in bbox_preds_ss
                ]
                flatten_angle_preds_ss = [
                    angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
                    for angle_pred in angle_preds_ss
                ]
                flatten_bbox_preds_ss = torch.cat(flatten_bbox_preds_ss)
                flatten_angle_preds_ss = torch.cat(flatten_angle_preds_ss)
                pos_bbox_preds_ss = flatten_bbox_preds_ss[pos_inds_ss]
                pos_angle_preds_ss = flatten_angle_preds_ss[pos_inds_ss]
                pos_points_ss = flatten_points[pos_inds_ss]

            bbox_coder = self.bbox_coder
            pos_decoded_angle_preds = self.angle_coder.decode(
                pos_angle_preds, keepdim=True)
            pos_bbox_preds = torch.cat(
                [pos_bbox_preds, pos_decoded_angle_preds], dim=-1)
            pos_bbox_targets = torch.cat([pos_bbox_targets, pos_angle_targets],
                                         dim=-1)

            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)

            if self.weak_supervised:
                loss_bbox = self.loss_bbox(
                    self.obb2xyxy(pos_decoded_bbox_preds),
                    self.obb2xyxy(pos_decoded_target_preds),
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)
            else:
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)

            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)

            if has_valid_ss:
                pos_bbox_preds_ss = torch.cat(
                    [pos_bbox_preds_ss, pos_angle_preds_ss], dim=-1)

                pos_decoded_bbox_preds_ss = bbox_coder.decode(
                    pos_points_ss, pos_bbox_preds_ss)

                _h, _w = self.crop_size
                _ctr = tf.new_tensor([[(_w - 1) / 2, (_h - 1) / 2]])

                _xy = pos_decoded_bbox_preds[pos_inds_ss_v, :2]
                _wh = pos_decoded_bbox_preds[pos_inds_ss_v, 2:4]
                pos_angle_targets_ss = pos_decoded_bbox_preds[pos_inds_ss_v,
                                                              4:] + rot

                _xy = (_xy - _ctr).matmul(tf.T) + _ctr

                if self.rotation_agnostic_classes:
                    pos_labels_ss = pos_labels[pos_inds_ss_v]
                    pos_angle_targets_ss = self._process_rotation_agnostic(
                        pos_angle_targets_ss, pos_labels_ss, dim=None)

                pos_decoded_target_preds_ss = torch.cat(
                    [_xy, _wh, pos_angle_targets_ss], dim=-1)

                pos_centerness_targets_ss = pos_centerness_targets[
                    pos_inds_ss_v]

                centerness_denorm_ss = max(
                    pos_centerness_targets_ss.sum().detach(), 1)

                loss_bbox_ss = self.loss_bbox_ss(
                    pos_decoded_bbox_preds_ss,
                    pos_decoded_target_preds_ss,
                    weight=pos_centerness_targets_ss,
                    avg_factor=centerness_denorm_ss)

            else:
                loss_bbox_ss = pos_bbox_preds[[]].sum()

        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_bbox_ss = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_bbox_ss=loss_bbox_ss)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.
        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angle for each scale level
                with shape (N, num_points * encode_size, H, W)
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            angle_pred_list (list[Tensor]): Box angle for a single scale
                level with shape (N, num_points * encode_size, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            # dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        results = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        if self.square_classes:
            bboxes = get_box_tensor(results.bboxes)
            for id in self.square_classes:
                inds = results.labels == id
                bboxes[inds, :] = hbox2rbox(rbox2hbox(bboxes[inds, :]))

            results.bboxes = RotatedBoxes(bboxes)
        return results
