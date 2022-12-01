# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, images_to_levels,
                                multi_apply, select_single_mlvl, unmap)
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class AngleBranchRetinaHead(RetinaHead):
    """retina head with angle regression branch.

    The head contains three subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors, the third regresses angles.

    Args:
        use_encoded_angle (:obj:`ConfigDict` or dict): Decide whether to use
            encoded angle or gt angle as target. Defaults to True.
        shield_reg_angle (:obj:`ConfigDict` or dict): Decide whether to shield
            the angle loss from reg branch. Defaults to False.
        angle_coder (dict): Config of angle coder.
        loss_angle (dict): Config of angle classification loss.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 *args,
                 use_encoded_angle: bool = True,
                 shield_reg_angle: bool = False,
                 use_normalized_angle_feat: bool = False,
                 angle_coder: ConfigType = dict(
                     type='CSLCoder',
                     angle_version='le90',
                     omega=1,
                     window='gaussian',
                     radius=6),
                 loss_angle: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='retina_cls',
                             std=0.01,
                             bias_prob=0.01),
                         dict(
                             type='Normal',
                             name='retina_angle_cls',
                             std=0.01,
                             bias_prob=0.01),
                     ]),
                 **kwargs) -> None:
        self.angle_coder = TASK_UTILS.build(angle_coder)
        self.encode_size = self.angle_coder.encode_size
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self.loss_angle = MODELS.build(loss_angle)
        self.shield_reg_angle = shield_reg_angle
        self.use_encoded_angle = use_encoded_angle
        self.use_normalized_angle_feat = use_normalized_angle_feat

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.retina_angle_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.encode_size,
            3,
            padding=1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:

            - cls_score (Tensor): Cls scores for a single scale level
              the channels number is num_anchors * num_classes.
            - bbox_pred (Tensor): Box energies / deltas for a single scale
              level, the channels number is num_anchors * 5.
            - angle_pred (Tensor): Angle for a single scale level the channels
              number is num_anchors * encode_size.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        angle_pred = self.retina_angle_cls(reg_feat)
        if self.use_normalized_angle_feat:
            angle_pred = angle_pred.sigmoid() * 2 - 1
        return cls_score, bbox_pred, angle_pred

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, anchors: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, bbox_weights: Tensor,
                            angle_targets: Tensor, angle_weights: Tensor,
                            avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            angle_pred (Tensor): Box angles for each scale
                level with shape (N, num_anchors * encode_size, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 5).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 5).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 5).
            angle_targets (Tensor): Angle regression targets of each anchor
                weight shape (N, num_total_anchors, 1).
            angle_weights (Tensor): Angle regression loss weights of each
                anchor with shape (N, num_total_anchors, 1).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # Equivalent substitution of ``@force_fp32()``
        cls_score = cls_score.float()
        bbox_pred = bbox_pred.float()

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        # Shield angle in reg. branch
        if self.shield_reg_angle:
            bbox_weights[:, -1] = 0.
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        angle_pred = angle_pred.permute(0, 2, 3,
                                        1).reshape(-1, self.encode_size)
        angle_targets = angle_targets.reshape(-1, self.encode_size)
        angle_weights = angle_weights.reshape(-1, 1)

        loss_angle = self.loss_angle(
            angle_pred,
            angle_targets,
            weight=angle_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_angle

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            angle_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            angle_preds (list[Tensor]): Box angles for each scale
                level with shape (N, num_anchors * encode_size, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, angel_target_list, angel_weight_list) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_angle = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            angle_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            angel_target_list,
            angel_weight_list,
            avg_factor=avg_factor)
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_angle=losses_angle)

    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor or :obj:`BaseBoxes`): Multi-level anchors
                of the image, which are concatenated into a single tensor
                or box type of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.

        Returns:
            tuple:

            - labels (Tensor): Labels of each level.
            - label_weights (Tensor): Label weights of each level.
            - bbox_targets (Tensor): BBox targets of each level.
            - bbox_weights (Tensor): BBox weights of each level.
            - pos_inds (Tensor): positive samples indexes.
            - neg_inds (Tensor): negative samples indexes.
            - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)
        angle_targets = anchors.new_zeros(num_valid_anchors, self.encode_size)
        angle_weights = anchors.new_zeros(num_valid_anchors, 1)

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

            angle_targets = anchors.new_zeros(num_valid_anchors, 1)
            if self.use_encoded_angle:
                # Get encoded angle as target
                angle_targets[pos_inds, :] = pos_bbox_targets[:, 4:5]
            else:
                # Get gt angle as target
                angle_targets[pos_inds, :] = \
                    sampling_result.pos_gt_bboxes[:, 4:5]
            # Angle encoder
            angle_targets = self.angle_coder.encode(angle_targets)
            angle_weights[pos_inds, :] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            angle_targets = unmap(angle_targets, num_total_anchors,
                                  inside_flags)
            angle_weights = unmap(angle_weights, num_total_anchors,
                                  inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, angle_targets, angle_weights)

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
            angle_preds (list[Tensor]): Box angles for each scale
                level with shape (N, num_anchors * encode_size, H, W)
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
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
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
            angle_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * encode_size, H, W).
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
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
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
        for idx, (cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            # Equivalent substitution of ``@force_fp32()``
            cls_score = cls_score.float()
            bbox_pred = bbox_pred.float()

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            angle_pred = angle_pred.permute(1, 2,
                                            0).reshape(-1, self.encode_size)
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
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            angle_pred = angle_pred[keep_idxs]

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            # Angle decoder
            angle_pred = self.angle_coder.decode(angle_pred)

            if self.use_encoded_angle:
                bbox_pred[..., -1] = angle_pred
                bbox_pred = self.bbox_coder.decode(
                    priors, bbox_pred, max_shape=img_shape)
            else:
                bbox_pred = self.bbox_coder.decode(
                    priors, bbox_pred, max_shape=img_shape)
                bbox_pred[..., -1] = angle_pred

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bboxes = cat_boxes(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)

        results = InstanceData()
        results.bboxes = bboxes
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
