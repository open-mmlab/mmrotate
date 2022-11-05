# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.layers import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances, multi_apply
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import ConfigType, InstanceList, OptMultiConfig
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures.bbox import QuadriBoxes, hbox2qbox


@MODELS.register_module()
class GVBBoxHead(BaseModule):
    """Gliding Vertex's  RoI head, with only two fc layers for classification
    and regression respectively."""

    def __init__(self,
                 with_avg_pool: bool = False,
                 num_shared_fcs: int = 0,
                 with_cls: bool = True,
                 with_reg: bool = True,
                 roi_feat_size: int = 7,
                 in_channels: int = 256,
                 fc_out_channels: int = 1024,
                 num_classes: int = 80,
                 ratio_thr: float = 0.8,
                 bbox_coder: ConfigType = dict(
                     type='DeltaXYWHQBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 fix_coder: ConfigType = dict(type='GVFixCoder'),
                 ratio_coder: ConfigType = dict(type='GVRatioCoder'),
                 predict_box_type: str = 'qbox',
                 reg_class_agnostic: bool = False,
                 reg_decoded_bbox: bool = False,
                 reg_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
                 cls_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
                 fix_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
                 ratio_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_fix: ConfigType = dict(
                     type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_ratio: ConfigType = dict(
                     type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.num_shared_fcs = num_shared_fcs
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.ratio_thr = ratio_thr
        self.predict_box_type = predict_box_type
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fix_predictor_cfg = fix_predictor_cfg
        self.ratio_predictor_cfg = ratio_predictor_cfg

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.fix_coder = TASK_UTILS.build(fix_coder)
        self.ratio_coder = TASK_UTILS.build(ratio_coder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_fix = MODELS.build(loss_fix)
        self.loss_ratio = MODELS.build(loss_ratio)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        self.shared_fcs = nn.ModuleList()
        for i in range(self.num_shared_fcs):
            fc_in_channels = (in_channels if i == 0 else self.fc_out_channels)
            self.shared_fcs.append(
                nn.Linear(fc_in_channels, self.fc_out_channels))

        last_dim = in_channels if self.num_shared_fcs == 0 \
            else self.fc_out_channels

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            # need to add background class
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=last_dim, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if reg_class_agnostic else \
                box_dim * num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            reg_predictor_cfg_.update(
                in_features=last_dim, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)
            out_dim_fix = 4 if reg_class_agnostic else \
                4 * num_classes
            fix_predictor_cfg_ = self.reg_predictor_cfg.copy()
            fix_predictor_cfg_.update(
                in_features=last_dim, out_features=out_dim_fix)
            self.fc_fix = MODELS.build(fix_predictor_cfg_)
            out_dim_ratio = 1 if reg_class_agnostic else \
                1 * num_classes
            ratio_predictor_cfg_ = self.reg_predictor_cfg.copy()
            ratio_predictor_cfg_.update(
                in_features=last_dim, out_features=out_dim_ratio)
            self.fc_ratio = MODELS.build(ratio_predictor_cfg_)

        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.num_shared_fcs > 0:
                self.init_cfg += [
                    dict(
                        type='Xavier',
                        distribution='uniform',
                        override=dict(name='shared_fcs'))
                ]
            if self.with_cls:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.01, override=dict(name='fc_cls'))
                ]
            if self.with_reg:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.001, override=dict(name='fc_reg'))
                ]
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.001, override=dict(name='fc_fix'))
                ]
                self.init_cfg += [
                    dict(
                        type='Normal',
                        std=0.001,
                        override=dict(name='fc_ratio'))
                ]

    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_cls_channels(self) -> bool:
        """get custom_cls_channels from loss_cls."""
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_activation(self) -> bool:
        """get custom_activation from loss_cls."""
        return getattr(self.loss_cls, 'custom_activation', False)

    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_accuracy(self) -> bool:
        """get custom_accuracy from loss_cls."""
        return getattr(self.loss_cls, 'custom_accuracy', False)

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

            - cls_score (Tensor): Classification scores for all
              scale levels, each is a 4D-tensor, the channels number
              is num_base_priors * num_classes.
            - bbox_pred (Tensor): Box energies / deltas for all
              scale levels, each is a 4D-tensor, the channels number
              is num_base_priors * 4.
            - fix_pred (Tensor): Fix / deltas for all
              scale levels, each is a 4D-tensor, the channels number
              is num_base_priors * 4.
            - ratio_pred (Tensor): Ratio / deltas for all
              scale levels, each is a 4D-tensor, the channels number
              is num_base_priors * 4.
        """
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))

        x = x.view(x.size(0), -1)

        if self.num_shared_fcs > 0:
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        fix_pred = torch.sigmoid(self.fc_fix(x)) if self.with_reg else None
        ratio_pred = torch.sigmoid(self.fc_ratio(x)) if self.with_reg else None

        return cls_score, bbox_pred, fix_pred, ratio_pred

    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

            - labels(Tensor): Gt_labels for all proposals, has
              shape (num_proposals,).
            - label_weights(Tensor): Labels_weights for all
              proposals, has shape (num_proposals,).
            - bbox_targets(Tensor):Regression target for all
              proposals, has shape (num_proposals, 4), the
              last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            - bbox_weights(Tensor):Regression weights for all
              proposals, has shape (num_proposals, 4).
            - fix_targets(Tensor):Fix target for all
              proposals, has shape (num_proposals, 4).
            - fix_weights(Tensor):Fix weights for all
              proposals, has shape (num_proposals, 4).
            - ratio_targets(Tensor):Ratio target for all
              proposals, has shape (num_proposals, 1).
            - ratio_weights(Tensor):Ratio weights for all
              proposals, has shape (num_proposals, 1).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        fix_targets = pos_priors.new_zeros(num_samples, 4)
        fix_weights = pos_priors.new_zeros(num_samples, 4)
        ratio_targets = pos_priors.new_zeros(num_samples, 1)
        ratio_weights = pos_priors.new_zeros(num_samples, 1)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
                pos_fix_targets = self.fix_coder.encode(pos_gt_bboxes)
                pos_ratio_targets = self.ratio_coder.encode(pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
            fix_targets[:num_pos, :] = pos_fix_targets
            fix_weights[:num_pos, :] = 1
            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
                fix_weights, ratio_targets, ratio_weights)

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
              proposals in a batch, each tensor in list has
              shape (num_proposals,) when `concat=False`, otherwise
              just a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
              all proposals in a batch, each tensor in list has
              shape (num_proposals,) when `concat=False`, otherwise
              just a single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
              for all proposals in a batch, each tensor in list
              has shape (num_proposals, 4) when `concat=False`,
              otherwise just a single tensor has shape
              (num_all_proposals, 4), the last dimension 4 represents
              [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals, 4) when `concat=False`, otherwise just a
              single tensor has shape (num_all_proposals, 4).
        """
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
         fix_weights, ratio_targets, ratio_weights) = multi_apply(
             self._get_targets_single,
             pos_priors_list,
             neg_priors_list,
             pos_gt_bboxes_list,
             pos_gt_labels_list,
             cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            fix_targets = torch.cat(fix_targets, 0)
            fix_weights = torch.cat(fix_weights, 0)
            ratio_targets = torch.cat(ratio_targets, 0)
            ratio_weights = torch.cat(ratio_weights, 0)
        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
                fix_weights, ratio_targets, ratio_weights)

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        fix_pred: Tensor,
                        ratio_pred: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            fix_pred (Tensor): Fix prediction results,
                has shape (batch_size * num_proposals_single_image, 4).
            ratio_pred (Tensor): Ratio prediction results,
                has shape (batch_size * num_proposals_single_image, 1).
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
            The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        losses = self.loss(
            cls_score,
            bbox_pred,
            fix_pred,
            ratio_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             fix_pred: Tensor,
             ratio_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             fix_targets: Tensor,
             fix_weights: Tensor,
             ratio_targets: Tensor,
             ratio_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            fix_pred (Tensor): Fix prediction results,
                has shape (batch_size * num_proposals_single_image, 4).
            ratio_pred (Tensor): Ratio prediction results,
                has shape (batch_size * num_proposals_single_image, 1).
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            fix_targets (Tensor): Fix target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            fix_weights (Tensor): Fix weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 1).
            ratio_targets (Tensor): Ratio target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            ratio_weights (Tensor): Ratio weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 1).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                    pos_fix_pred = fix_pred.view(fix_pred.size(0),
                                                 4)[pos_inds.type(torch.bool)]
                    pos_ratio_pred = ratio_pred.view(
                        ratio_pred.size(0), 1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                    pos_fix_pred = fix_pred.view(
                        fix_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_ratio_pred = ratio_pred.view(
                        ratio_pred.size(0), -1,
                        1)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)

                losses['loss_fix'] = self.loss_bbox(
                    pos_fix_pred,
                    fix_targets[pos_inds.type(torch.bool)],
                    fix_weights[pos_inds.type(torch.bool)],
                    avg_factor=fix_targets.size(0),
                    reduction_override=reduction_override)
                losses['loss_ratio'] = self.loss_bbox(
                    pos_ratio_pred,
                    ratio_targets[pos_inds.type(torch.bool)],
                    ratio_weights[pos_inds.type(torch.bool)],
                    avg_factor=ratio_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
                losses['loss_fix'] = fix_pred[pos_inds].sum()
                losses['loss_ratio'] = ratio_pred[pos_inds].sum()

        return losses

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        fix_preds: Tuple[Tensor],
                        ratio_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            fix_preds (tuple[Tensor]): Tuple of fix / deltas, each
                has shape (num_boxes, num_classes * 4).
            ratio_preds (tuple[Tensor]): Tuple of ratio / deltas, each
                has shape (num_boxes, num_classes * 1).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 8),
              the last dimension 4 arrange as (x1, y1, ..., x4, y4).
        """
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                fix_pred=fix_preds[img_id],
                ratio_pred=ratio_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            fix_pred: Tensor,
            ratio_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            fix_pred (Tensor): Fix / deltas.
                has shape (num_boxes, num_classes * 4).
            ratio_pred (Tensor): Ratio / deltas.
                has shape (num_boxes, num_classes * 1).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 8),
              the last dimension 4 arrange as (x1, y1, ..., x4, y4).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        qboxes = self.fix_coder.decode(bboxes, fix_pred)
        bboxes = bboxes.view(*ratio_pred.size(), 4)
        qboxes = qboxes.view(*ratio_pred.size(), 8)

        qboxes[ratio_pred > self.ratio_thr] = \
            hbox2qbox(bboxes[ratio_pred > self.ratio_thr])
        bboxes = QuadriBoxes(qboxes)

        if self.predict_box_type == 'rbox':
            bboxes = bboxes.detach().convert_to('rbox')

        if rescale and qboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(qboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
