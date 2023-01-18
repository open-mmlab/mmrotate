# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
from mmcv.cnn import ConvModule, Scale, is_norm
from mmdet.models import inverse_sigmoid
from mmdet.models.dense_heads import RTMDetHead
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap)
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, cat_boxes, distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, distance2obb


@MODELS.register_module()
class RotatedRTMDetHead(RTMDetHead):
    """Detection Head of Rotated RTMDet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        angle_version (str): Angle representations. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Default to False.
        scale_angle (bool): If true, add scale to angle pred branch.
            Default to True.
        angle_coder (:obj:`ConfigDict` or dict): Config of angle coder.
        loss_angle (:obj:`ConfigDict` or dict, Optional): Config of angle loss.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 angle_version: str = 'le90',
                 use_hbbox_loss: bool = False,
                 scale_angle: bool = True,
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 loss_angle: OptConfigType = None,
                 **kwargs) -> None:
        self.angle_version = angle_version
        self.use_hbbox_loss = use_hbbox_loss
        self.is_scale_angle = scale_angle
        self.angle_coder = TASK_UTILS.build(angle_coder)
        super().__init__(
            num_classes,
            in_channels,
            # useless, but error
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            **kwargs)
        if loss_angle is not None:
            self.loss_angle = MODELS.build(loss_angle)
        else:
            self.loss_angle = None

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        pred_pad_size = self.pred_kernel_size // 2
        self.rtm_ang = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.angle_coder.encode_size,
            self.pred_kernel_size,
            padding=pred_pad_size)
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.rtm_ang, std=0.01)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_dim.
        """

        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj(reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))

            reg_dist = scale(self.rtm_reg(reg_feat).exp()).float() * stride[0]
            if self.is_scale_angle:
                angle_pred = self.scale_angle(self.rtm_ang(reg_feat)).float()
            else:
                angle_pred = self.rtm_ang(reg_feat).float()

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            assign_metrics: Tensor, stride: List[int]):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 5, H, W) for rbox loss
                or (N, num_anchors * 4, H, W) for hbox loss.
            angle_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape
                (N, num_total_anchors).
            stride (List[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()

        if self.use_hbbox_loss:
            bbox_pred = bbox_pred.reshape(-1, 4)
        else:
            bbox_pred = bbox_pred.reshape(-1, 5)
        bbox_targets = bbox_targets.reshape(-1, 5)

        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets
            if self.use_hbbox_loss:
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(
                    pos_bbox_targets[:, :4])

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_angle = angle_pred.sum() * 0
            if self.loss_angle is not None:
                angle_pred = angle_pred.reshape(-1,
                                                self.angle_coder.encode_size)
                pos_angle_pred = angle_pred[pos_inds]
                pos_angle_target = pos_bbox_targets[:, 4:5]
                pos_angle_target = self.angle_coder.encode(pos_angle_target)
                if pos_angle_target.dim() == 2:
                    pos_angle_weight = pos_bbox_weight.unsqueeze(-1)
                else:
                    pos_angle_weight = pos_bbox_weight
                loss_angle = self.loss_angle(
                    pos_angle_pred,
                    pos_angle_target,
                    weight=pos_angle_weight,
                    avg_factor=1.0)

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)

        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)
            loss_angle = angle_pred.sum() * 0

        return (loss_cls, loss_bbox, loss_angle, assign_metrics.sum(),
                pos_bbox_weight.sum(), pos_bbox_weight.sum())

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box predict for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [t, b, l, r] format.
            bbox_preds (list[Tensor]): Angle pred for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
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
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)

        decoded_bboxes = []
        decoded_hbboxes = []
        angle_preds_list = []
        for anchor, bbox_pred, angle_pred in zip(anchor_list[0], bbox_preds,
                                                 angle_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.angle_coder.encode_size)

            if self.use_hbbox_loss:
                hbbox_pred = distance2bbox(anchor, bbox_pred)
                decoded_hbboxes.append(hbbox_pred)

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            bbox_pred = distance2obb(
                anchor, bbox_pred, angle_version=self.angle_version)
            decoded_bboxes.append(bbox_pred)
            angle_preds_list.append(angle_pred)

        # flatten_bboxes is rbox, for target assign
        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets

        if self.use_hbbox_loss:
            decoded_bboxes = decoded_hbboxes

        (losses_cls, losses_bbox, losses_angle, cls_avg_factors,
         bbox_avg_factors, angle_avg_factors) = multi_apply(
             self.loss_by_feat_single, cls_scores, decoded_bboxes,
             angle_preds_list, labels_list, label_weights_list,
             bbox_targets_list, assign_metrics_list,
             self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        if self.loss_angle is not None:
            angle_avg_factors = reduce_mean(
                sum(angle_avg_factors)).clamp_(min=1).item()
            losses_angle = list(
                map(lambda x: x / angle_avg_factors, losses_angle))
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_angle=losses_angle)
        else:
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.

            - anchors (Tensor): All anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with shape
              (N,).
            - label_weights (Tensor): Label weights of all anchor in the
              image with shape (N,).
            - bbox_targets (Tensor): BBox targets of all anchors in the
              image with shape (N, 5).
            - norm_alignment_metrics (Tensor): Normalized alignment metrics
              of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :],
            priors=anchors)

        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((*anchors.size()[:-1], 5))
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            pos_bbox_targets = pos_bbox_targets.regularize_boxes(
                self.angle_version)
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[
                gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors,
                                   inside_flags)
        return (anchors, labels, label_weights, bbox_targets, assign_metrics,
                sampling_result)

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
                with shape (N, num_points * angle_dim, H, W)
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
                level with shape (N, num_points * angle_dim, H, W).
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

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)


@MODELS.register_module()
class RotatedRTMDetSepBNHead(RotatedRTMDetHead):
    """Rotated RTMDetHead with separated BN layers and shared conv layers.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        scale_angle (bool): Does not support in RotatedRTMDetSepBNHead,
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict)): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict)): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
        exp_on_reg (bool): Whether to apply exponential on bbox_pred.
            Defaults to False.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 share_conv: bool = True,
                 scale_angle: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 pred_kernel_size: int = 1,
                 exp_on_reg: bool = False,
                 **kwargs) -> None:
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        assert scale_angle is False, \
            'scale_angle does not support in RotatedRTMDetSepBNHead'
        super().__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            pred_kernel_size=pred_kernel_size,
            scale_angle=False,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_ang = nn.ModuleList()
        if self.with_objectness:
            self.rtm_obj = nn.ModuleList()
        for n in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_ang.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.angle_coder.encode_size,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(
                        self.feat_channels,
                        1,
                        self.pred_kernel_size,
                        padding=self.pred_kernel_size // 2))

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg, rtm_ang in zip(self.rtm_cls, self.rtm_reg,
                                             self.rtm_ang):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)
            normal_init(rtm_ang, std=0.01)
        if self.with_objectness:
            for rtm_obj in self.rtm_obj:
                normal_init(rtm_obj, std=0.01, bias=bias_cls)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_dim.
        """
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            angle_pred = self.rtm_ang[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)
