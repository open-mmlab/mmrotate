# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import chamfer_distance, min_area_polygons
from mmdet.models.utils import images_to_levels, multi_apply, unmap
from mmdet.utils import ConfigType, InstanceList, OptInstanceList
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.models.dense_heads.rotated_reppoints_head import \
    RotatedRepPointsHead
from mmrotate.registry import MODELS
from ..utils import levels_to_images


def ChamferDistance2D(point_set_1: Tensor,
                      point_set_2: Tensor,
                      distance_weight: float = 0.05,
                      eps: float = 1e-12):
    """Compute the Chamfer distance between two point sets.

    Args:
        point_set_1 (Tensor): point set 1 with shape
            (N_pointsets, N_points, 2)
        point_set_2 (Tensor): point set 2 with shape
            (N_pointsets, N_points, 2)
        distance_weight (float): weight of chamfer distance loss.
        eps (float): a value added to the denominator for numerical
            stability. Defaults to 1e-12.

    Returns:
        Tensor: chamfer distance between two point sets
        with shape (N_pointsets,)
    """
    assert point_set_1.dim() == point_set_2.dim()
    assert point_set_1.shape[-1] == point_set_2.shape[-1]
    assert point_set_1.dim() <= 3
    dist1, dist2, _, _ = chamfer_distance(point_set_1, point_set_2)
    dist1 = torch.sqrt(torch.clamp(dist1, eps))
    dist2 = torch.sqrt(torch.clamp(dist2, eps))
    dist = distance_weight * (dist1.mean(-1) + dist2.mean(-1)) / 2.0

    return dist


@MODELS.register_module()
class OrientedRepPointsHead(RotatedRepPointsHead):
    """Oriented RepPoints head -<https://arxiv.org/pdf/2105.11111v4.pdf>. The
    head contains initial and refined stages based on RepPoints. The initial
    stage regresses coarse point sets, and the refine stage further regresses
    the fine point sets. The APAA scheme based on the quality of point set
    samples in the paper is employed in refined stage.

    Args:
        loss_spatial_init  (:obj:`ConfigDict` or dict): Config of initial
            spatial loss.
        loss_spatial_refine  (:obj:`ConfigDict` or dict): Config of refine
            spatial loss.
        top_ratio (float): Ratio of top high-quality point sets.
            Defaults to 0.4.
        init_qua_weight (float): Quality weight of initial stage.
            Defaults to 0.2.
        ori_qua_weight (float): Orientation quality weight.
            Defaults to 0.3.
        poc_qua_weight (float): Point-wise correlation quality weight.
            Defaults to 0.1.
    """  # noqa: W605

    def __init__(self,
                 *args,
                 loss_spatial_init: ConfigType = dict(
                     type='SpatialBorderLoss', loss_weight=0.05),
                 loss_spatial_refine: ConfigType = dict(
                     type='SpatialBorderLoss', loss_weight=0.1),
                 top_ratio: float = 0.4,
                 init_qua_weight: float = 0.2,
                 ori_qua_weight: float = 0.3,
                 poc_qua_weight: float = 0.1,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_spatial_init = MODELS.build(loss_spatial_init)
        self.loss_spatial_refine = MODELS.build(loss_spatial_refine)
        self.top_ratio = top_ratio
        self.init_qua_weight = init_qua_weight
        self.ori_qua_weight = ori_qua_weight
        self.poc_qua_weight = poc_qua_weight

    def forward_single(self, x: Tensor) -> Tuple[Tensor]:
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        base_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        if self.training:
            return cls_out, pts_out_init, pts_out_refine, base_feat
        else:
            return cls_out, pts_out_refine

    def _get_targets_single(self,
                            flat_proposals: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            gt_instances_ignore: InstanceData,
                            stage: str = 'init',
                            unmap_outputs: bool = True) -> tuple:
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            flat_proposals (Tensor): Multi level points of a image.
            valid_flags (Tensor): Multi level valid flags of a image.
            gt_instances (InstanceData): It usually includes ``bboxes`` and
                ``labels`` attributes.
            gt_instances_ignore (InstanceData): It includes ``bboxes``
                attribute data that is ignored during training and testing.
            stage (str): 'init' or 'refine'. Generate target for
                init stage or refine stage. Defaults to 'init'.
            unmap_outputs (bool): Whether to map outputs back to
                the original set of anchors. Defaults to True.

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
        inside_flags = valid_flags
        if not inside_flags.any():
            raise ValueError(
                'There is no valid proposal inside the image boundary. Please '
                'check the image size.')
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]
        pred_instances = InstanceData(priors=proposals)

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight

        assign_result = assigner.assign(pred_instances, gt_instances,
                                        gt_instances_ignore)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        gt_inds = assign_result.gt_inds
        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 8])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros(num_valid_proposals)
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            bbox_gt[pos_inds, :] = sampling_result.pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(
                labels,
                num_total_proposals,
                inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)
            gt_inds = unmap(gt_inds, num_total_proposals, inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, gt_inds,
                sampling_result)

    def get_targets(self,
                    proposals_list: List[Tensor],
                    valid_flag_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    stage: str = 'init',
                    unmap_outputs: bool = True) -> tuple:
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[Tensor]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[Tensor]): Multi level valid flags of each
                image.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            stage (str): 'init' or 'refine'. Generate target for init stage or
                refine stage.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:

            - labels_list (list[Tensor]): Labels of each level.
            - label_weights_list (list[Tensor]): Label weights of each
              level.
            - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
            - proposals_list (list[Tensor]): Proposals(points/bboxes) of
              each level.
            - proposal_weights_list (list[Tensor]): Proposal weights of
              each level.
            - avg_factor (int): Average factor that is used to average
              the loss. When using sampling method, avg_factor is usually
              the sum of positive and negative priors. When using
              `PseudoSampler`, `avg_factor` is usually equal to the number
              of positive priors.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(batch_img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list, all_gt_inds,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             proposals_list,
             valid_flag_list,
             batch_gt_instances,
             batch_gt_instances_ignore,
             stage=stage,
             unmap_outputs=unmap_outputs)

        if stage == 'init':
            # no valid points
            if any([labels is None for labels in all_labels]):
                return None
            # sampled points of all images
            num_total_pos = sum(
                [max(inds.numel(), 1) for inds in pos_inds_list])
            num_total_neg = sum(
                [max(inds.numel(), 1) for inds in neg_inds_list])
            # avg_refactor = sum(
            #     [results.avg_factor for results in sampling_results_list])
            labels_list = images_to_levels(all_labels, num_level_proposals)
            label_weights_list = images_to_levels(all_label_weights,
                                                  num_level_proposals)
            bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
            proposals_list = images_to_levels(all_proposals,
                                              num_level_proposals)
            proposal_weights_list = images_to_levels(all_proposal_weights,
                                                     num_level_proposals)
            res = (labels_list, label_weights_list, bbox_gt_list,
                   proposals_list, proposal_weights_list, num_total_pos,
                   num_total_neg)
        else:
            pos_inds = []
            pos_gt_index = []
            for i, single_labels in enumerate(all_labels):
                pos_mask = (0 <= single_labels) & (
                    single_labels < self.num_classes)
                pos_inds.append(pos_mask.nonzero(as_tuple=False).view(-1))
                pos_gt_index.append(
                    all_gt_inds[i][pos_mask.nonzero(as_tuple=False).view(-1)])
            res = (all_labels, all_label_weights, all_bbox_gt, all_proposals,
                   all_proposal_weights, pos_inds, pos_gt_index)

        return res

    def loss_by_feat_single(self, cls_score: Tensor, pts_pred_init: Tensor,
                            pts_pred_refine: Tensor, labels: Tensor,
                            label_weights, bbox_gt_init: Tensor,
                            bbox_weights_init: Tensor, bbox_gt_refine: Tensor,
                            bbox_weights_refine: Tensor, stride: int,
                            avg_factor_init: int,
                            avg_factor_refine: int) -> Tuple[Tensor]:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_classes, h_i, w_i).
            pts_pred_init (Tensor): Points of shape
                (batch_size, h_i * w_i, num_points * 2).
            pts_pred_refine (Tensor): Points refined of shape
                (batch_size, h_i * w_i, num_points * 2).
            labels (Tensor): Ground truth class indices with shape
                (batch_size, h_i * w_i).
            label_weights (Tensor): Label weights of shape
                (batch_size, h_i * w_i).
            bbox_gt_init (Tensor): BBox regression targets in the init stage
                of shape (batch_size, h_i * w_i, 8).
            bbox_weights_init (Tensor): BBox regression loss weights in the
                init stage of shape (batch_size, h_i * w_i, 8).
            bbox_gt_refine (Tensor): BBox regression targets in the refine
                stage of shape (batch_size, h_i * w_i, 8).
            bbox_weights_refine (Tensor): BBox regression loss weights in the
                refine stage of shape (batch_size, h_i * w_i, 8).
            stride (int): Point stride.
            avg_factor_init (int): Average factor that is used to average
                the loss in the init stage.
            avg_factor_refine (int): Average factor that is used to average
                the loss in the refine stage.

        Returns:
            Tuple[Tensor]: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score.contiguous()
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor_refine)

        # init loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 8)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        bbox_weights_init = bbox_weights_init.reshape(-1)
        pos_ind_init = (bbox_weights_init > 0).nonzero(
            as_tuple=False).reshape(-1)
        pos_bbox_gt_init = bbox_gt_init[pos_ind_init]
        pos_pts_pred_init = pts_pred_init[pos_ind_init]
        pos_bbox_weights_init = bbox_weights_init[pos_ind_init]
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(
            pos_pts_pred_init / normalize_term,
            pos_bbox_gt_init / normalize_term,
            pos_bbox_weights_init,
            avg_factor=avg_factor_init)

        # refine loss
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 8)
        pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        bbox_weights_refine = bbox_weights_refine.reshape(-1)
        pos_ind_refine = (bbox_weights_refine > 0).nonzero(
            as_tuple=False).reshape(-1)
        pos_bbox_gt_refine = bbox_gt_refine[pos_ind_refine]
        pos_pts_pred_refine = pts_pred_refine[pos_ind_refine]
        pos_bbox_weights_refine = bbox_weights_refine[pos_ind_refine]
        loss_pts_refine = self.loss_bbox_refine(
            pos_pts_pred_refine / normalize_term,
            pos_bbox_gt_refine / normalize_term,
            pos_bbox_weights_refine,
            avg_factor=avg_factor_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        pts_preds_init: List[Tensor],
        pts_preds_refine: List[Tensor],
        base_feat: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, of shape (batch_size, num_classes, h, w).
            pts_preds_init (list[Tensor]): Points for each scale level, each is
                a 3D-tensor, of shape (batch_size, h_i * w_i, num_points * 2).
            pts_preds_refine (list[Tensor]): Points refined for each scale
                level, each is a 3D-tensor, of shape
                (batch_size, h_i * w_i, num_points * 2).
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
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       batch_img_metas, device)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)

        num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                    for featmap in cls_scores]

        num_level = len(featmap_sizes)
        assert num_level == len(pts_coordinate_preds_init)

        if self.train_cfg.init.assigner['type'] == 'ConvexAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            raise NotImplementedError
        cls_reg_targets_init = self.get_targets(
            proposals_list=candidate_list,
            valid_flag_list=valid_flag_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='init')
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       batch_img_metas, device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)

        refine_points_features, = multi_apply(self.get_adaptive_points_feature,
                                              base_feat,
                                              pts_coordinate_preds_refine,
                                              self.point_strides)
        features_pts_refine = levels_to_images(refine_points_features)
        features_pts_refine = [
            item.reshape(-1, self.num_points, item.shape[-1])
            for item in features_pts_refine
        ]

        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(
                    points_preds_init_.shape[0], -1,
                    *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(
                    0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                bbox.append(
                    points_center +
                    points_shift[i_img].reshape(-1, 2 * self.num_points))
            bbox_list.append(bbox)
        cls_reg_targets_refine = self.get_targets(
            proposals_list=bbox_list,
            valid_flag_list=valid_flag_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='refine')
        (labels_list, label_weights_list, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine, pos_inds_list_refine,
         pos_gt_index_list_refine) = cls_reg_targets_refine

        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]

        pts_coordinate_preds_init_img = levels_to_images(
            pts_coordinate_preds_init, flatten=True)
        pts_coordinate_preds_init_img = [
            item.reshape(-1, 2 * self.num_points)
            for item in pts_coordinate_preds_init_img
        ]

        pts_coordinate_preds_refine_img = levels_to_images(
            pts_coordinate_preds_refine, flatten=True)
        pts_coordinate_preds_refine_img = [
            item.reshape(-1, 2 * self.num_points)
            for item in pts_coordinate_preds_refine_img
        ]

        with torch.no_grad():

            quality_assess_list, = multi_apply(
                self.pointsets_quality_assessment, features_pts_refine,
                cls_scores, pts_coordinate_preds_init_img,
                pts_coordinate_preds_refine_img, labels_list,
                bbox_gt_list_refine, label_weights_list,
                bbox_weights_list_refine, pos_inds_list_refine)

            labels_list, label_weights_list, bbox_weights_list_refine, \
                num_pos, pos_normalize_term = multi_apply(
                    self.dynamic_pointset_samples_selection,
                    quality_assess_list,
                    labels_list,
                    label_weights_list,
                    bbox_weights_list_refine,
                    pos_inds_list_refine,
                    pos_gt_index_list_refine,
                    num_proposals_each_level=num_proposals_each_level,
                    num_level=num_level
                )
            num_pos = sum(num_pos)

        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        pts_preds_refine = torch.cat(pts_coordinate_preds_refine_img, 0).view(
            -1, pts_coordinate_preds_refine_img[0].size(-1))

        labels = torch.cat(labels_list, 0).view(-1)
        labels_weight = torch.cat(label_weights_list, 0).view(-1)
        bbox_gt_refine = torch.cat(bbox_gt_list_refine,
                                   0).view(-1, bbox_gt_list_refine[0].size(-1))
        bbox_weights_refine = torch.cat(bbox_weights_list_refine, 0).view(-1)
        pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)
        pos_inds_flatten = ((0 <= labels) &
                            (labels < self.num_classes)).nonzero(
                                as_tuple=False).reshape(-1)

        assert len(pos_normalize_term) == len(pos_inds_flatten)

        if num_pos:
            losses_cls = self.loss_cls(
                cls_scores, labels, labels_weight, avg_factor=num_pos)
            pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
            pos_bbox_gt_refine = bbox_gt_refine[pos_inds_flatten]

            pos_bbox_weights_refine = bbox_weights_refine[pos_inds_flatten]
            losses_pts_refine = self.loss_bbox_refine(
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                pos_bbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_bbox_weights_refine)

            loss_border_refine = self.loss_spatial_refine(
                pos_pts_pred_refine.reshape(-1, 2 * self.num_points) /
                pos_normalize_term.reshape(-1, 1),
                pos_bbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_bbox_weights_refine,
                avg_factor=None)

        else:
            losses_cls = cls_scores.sum() * 0
            losses_pts_refine = pts_preds_refine.sum() * 0
            loss_border_refine = pts_preds_refine.sum() * 0

        losses_pts_init, loss_border_init = multi_apply(
            self.init_loss_single, pts_coordinate_preds_init,
            bbox_gt_list_init, bbox_weights_list_init, self.point_strides)

        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine,
            'loss_spatial_init': loss_border_init,
            'loss_spatial_refine': loss_border_refine
        }
        return loss_dict_all

    def sampling_points(self, polygons: Tensor, points_num: int,
                        device: str) -> Tensor:
        """Sample edge points for polygon.

        Args:
            polygons (Tensor): polygons with shape (N, 8)
            points_num (int): number of sampling points for each polygon edge.
                10 by default.
            device (str): The device the tensor will be put on.
                Defaults to ``cuda``.

        Returns:
            sampling_points (Tensor): sampling points with shape (N,
            points_num*4, 2)
        """
        polygons_xs, polygons_ys = polygons[:, 0::2], polygons[:, 1::2]
        ratio = torch.linspace(0, 1, points_num).to(device).repeat(
            polygons.shape[0], 1)

        edge_pts_x = []
        edge_pts_y = []
        for i in range(4):
            if i < 3:
                points_x = ratio * polygons_xs[:, i + 1:i + 2] + (
                    1 - ratio) * polygons_xs[:, i:i + 1]
                points_y = ratio * polygons_ys[:, i + 1:i + 2] + (
                    1 - ratio) * polygons_ys[:, i:i + 1]
            else:
                points_x = ratio * polygons_xs[:, 0].unsqueeze(1) + (
                    1 - ratio) * polygons_xs[:, i].unsqueeze(1)
                points_y = ratio * polygons_ys[:, 0].unsqueeze(1) + (
                    1 - ratio) * polygons_ys[:, i].unsqueeze(1)

            edge_pts_x.append(points_x)
            edge_pts_y.append(points_y)

        sampling_points_x = torch.cat(edge_pts_x, dim=1).unsqueeze(dim=2)
        sampling_points_y = torch.cat(edge_pts_y, dim=1).unsqueeze(dim=2)
        sampling_points = torch.cat([sampling_points_x, sampling_points_y],
                                    dim=2)

        return sampling_points

    def get_adaptive_points_feature(self, features: Tensor,
                                    pt_locations: Tensor,
                                    stride: int) -> Tensor:
        """Get the points features from the locations of predicted points.

        Args:
            features (Tensor): base feature with shape (B,C,W,H)
            pt_locations (Tensor): locations of points in each point set
                with shape (B, N_points_set(number of point set),
                N_points(number of points in each point set) *2)
            stride (int): points strdie

        Returns:
            Tensor: sampling features with (B, C, N_points_set, N_points)
        """

        h = features.shape[2] * stride
        w = features.shape[3] * stride

        pt_locations = pt_locations.view(pt_locations.shape[0],
                                         pt_locations.shape[1], -1, 2).clone()
        pt_locations[..., 0] = pt_locations[..., 0] / (w / 2.) - 1
        pt_locations[..., 1] = pt_locations[..., 1] / (h / 2.) - 1

        batch_size = features.size(0)
        sampled_features = torch.zeros([
            pt_locations.shape[0],
            features.size(1),
            pt_locations.size(1),
            pt_locations.size(2)
        ]).to(pt_locations.device)

        for i in range(batch_size):
            feature = nn.functional.grid_sample(features[i:i + 1],
                                                pt_locations[i:i + 1])[0]
            sampled_features[i] = feature

        return sampled_features,

    def feature_cosine_similarity(self, points_features: Tensor) -> Tensor:
        """Compute the points features similarity for points-wise correlation.

        Args:
            points_features (Tensor): sampling point feature with
                shape (N_pointsets, N_points, C)

        Returns:
            max_correlation (Tensor): max feature similarity in each point set
            with shape (N_points_set, N_points, C)
        """

        mean_points_feats = torch.mean(points_features, dim=1, keepdim=True)
        norm_pts_feats = torch.norm(
            points_features, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)
        norm_mean_pts_feats = torch.norm(
            mean_points_feats, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)

        unity_points_features = points_features / norm_pts_feats
        unity_mean_points_feats = mean_points_feats / norm_mean_pts_feats

        feats_similarity = 1.0 - F.cosine_similarity(
            unity_points_features, unity_mean_points_feats, dim=2, eps=1e-6)

        max_correlation, _ = torch.max(feats_similarity, dim=1)

        return max_correlation

    def pointsets_quality_assessment(self, pts_features: Tensor,
                                     cls_score: Tensor, pts_pred_init: Tensor,
                                     pts_pred_refine: Tensor, label: Tensor,
                                     bbox_gt: Tensor, label_weight: Tensor,
                                     bbox_weight: Tensor,
                                     pos_inds: Tensor) -> Tensor:
        """Assess the quality of each point set from the classification,
        localization, orientation, and point-wise correlation based on the
        assigned point sets samples.

        Args:
            pts_features (Tensor): points features with shape (N, 9, C)
            cls_score (Tensor): classification scores with
                shape (N, class_num)
            pts_pred_init (Tensor): initial point sets prediction with
                shape (N, 9*2)
            pts_pred_refine (Tensor): refined point sets prediction with
                shape (N, 9*2)
            label (Tensor): gt label with shape (N)
            bbox_gt(Tensor): gt bbox of polygon with shape (N, 8)
            label_weight (Tensor): label weight with shape (N)
            bbox_weight (Tensor): box weight with shape (N)
            pos_inds (Tensor): the  inds of  positive point set samples

        Returns:
            qua (Tensor) : weighted quality values for positive
            point set samples.
        """
        device = cls_score.device

        # avoid no positive samplers
        if pos_inds.shape[0] == 0:
            pos_scores = cls_score
            pos_pts_pred_init = pts_pred_init
            pos_pts_pred_refine = pts_pred_refine
            pos_pts_refine_features = pts_features
            pos_bbox_gt = bbox_gt
            pos_label = label
            pos_label_weight = label_weight
            pos_bbox_weight = bbox_weight
        else:
            pos_scores = cls_score[pos_inds]
            pos_pts_pred_init = pts_pred_init[pos_inds]
            pos_pts_pred_refine = pts_pred_refine[pos_inds]
            pos_pts_refine_features = pts_features[pos_inds]
            pos_bbox_gt = bbox_gt[pos_inds]
            pos_label = label[pos_inds]
            pos_label_weight = label_weight[pos_inds]
            pos_bbox_weight = bbox_weight[pos_inds]

        # quality of point-wise correlation
        qua_poc = self.poc_qua_weight * self.feature_cosine_similarity(
            pos_pts_refine_features)

        qua_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        polygons_pred_init = min_area_polygons(pos_pts_pred_init)
        polygons_pred_refine = min_area_polygons(pos_pts_pred_refine)
        sampling_pts_pred_init = self.sampling_points(
            polygons_pred_init, 10, device=device)
        sampling_pts_pred_refine = self.sampling_points(
            polygons_pred_refine, 10, device=device)
        sampling_pts_gt = self.sampling_points(pos_bbox_gt, 10, device=device)

        # quality of orientation
        qua_ori_init = self.ori_qua_weight * ChamferDistance2D(
            sampling_pts_gt, sampling_pts_pred_init)
        qua_ori_refine = self.ori_qua_weight * ChamferDistance2D(
            sampling_pts_gt, sampling_pts_pred_refine)

        # quality of localization
        qua_loc_init = self.loss_bbox_refine(
            pos_pts_pred_init,
            pos_bbox_gt,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        qua_loc_refine = self.loss_bbox_refine(
            pos_pts_pred_refine,
            pos_bbox_gt,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        # quality of classification
        qua_cls = qua_cls.sum(-1)

        # weighted inti-stage and refine-stage
        qua = qua_cls + self.init_qua_weight * (
            qua_loc_init + qua_ori_init) + (1.0 - self.init_qua_weight) * (
                qua_loc_refine + qua_ori_refine) + qua_poc

        return qua,

    def dynamic_pointset_samples_selection(
            self,
            quality: Tensor,
            label: Tensor,
            label_weight: Tensor,
            bbox_weight: Tensor,
            pos_inds: Tensor,
            pos_gt_inds: Tensor,
            num_proposals_each_level: Optional[List[int]] = None,
            num_level: Optional[int] = None) -> tuple:
        """The dynamic top k selection of point set samples based on the
        quality assessment values.

        Args:
            quality (Tensor): the quality values of positive
                point set samples
            label (Tensor): gt label with shape (N)
            label_weight (Tensor): label weight with shape (N)
            bbox_weight (Tensor): box weight with shape (N)
            pos_inds (Tensor): the inds of  positive point set samples
            pos_gt_inds (Tensor): the inds of  positive ground truth
            num_proposals_each_level (list[int]): proposals number of
                each level
            num_level (int): the level number

        Returns:
            tuple:

            - label: gt label with shape (N)
            - label_weight: label weight with shape (N)
            - bbox_weight: box weight with shape (N)
            - num_pos (int): the number of selected positive point samples
              with high-quality
            - pos_normalize_term (Tensor): the corresponding positive
              normalize term
        """

        if len(pos_inds) == 0:
            return label, label_weight, bbox_weight, 0, Tensor(
                []).type_as(bbox_weight)

        num_gt = pos_gt_inds.max()
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)

        pos_inds_after_select = []
        ignore_inds_after_select = []

        for gt_ind in range(num_gt):
            pos_inds_select = []
            pos_loss_select = []
            gt_mask = pos_gt_inds == (gt_ind + 1)
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                value, topk_inds = quality[level_gt_mask].topk(
                    min(level_gt_mask.sum(), 6), largest=False)
                pos_inds_select.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_select.append(value)
            pos_inds_select = torch.cat(pos_inds_select)
            pos_loss_select = torch.cat(pos_loss_select)

            if len(pos_inds_select) < 2:
                pos_inds_after_select.append(pos_inds_select)
                ignore_inds_after_select.append(pos_inds_select.new_tensor([]))

            else:
                pos_loss_select, sort_inds = pos_loss_select.sort(
                )  # small to large
                pos_inds_select = pos_inds_select[sort_inds]
                # dynamic top k
                topk = math.ceil(pos_loss_select.shape[0] * self.top_ratio)
                pos_inds_select_topk = pos_inds_select[:topk]
                pos_inds_after_select.append(pos_inds_select_topk)
                ignore_inds_after_select.append(
                    pos_inds_select_topk.new_tensor([]))

        pos_inds_after_select = torch.cat(pos_inds_after_select)
        ignore_inds_after_select = torch.cat(ignore_inds_after_select)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_select).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_select] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_select)

        pos_level_mask_after_select = []
        for i in range(num_level):
            mask = (pos_inds_after_select >= inds_level_interval[i]) & (
                pos_inds_after_select < inds_level_interval[i + 1])
            pos_level_mask_after_select.append(mask)
        pos_level_mask_after_select = torch.stack(pos_level_mask_after_select,
                                                  0).type_as(label)
        pos_normalize_term = pos_level_mask_after_select * (
            self.point_base_scale *
            torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[
            pos_normalize_term > 0].type_as(bbox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_select)

        return label, label_weight, bbox_weight, num_pos, pos_normalize_term

    def init_loss_single(self, pts_pred_init: Tensor, bbox_gt_init: Tensor,
                         bbox_weights_init: Tensor,
                         stride: int) -> Tuple[Tensor, Tensor]:
        """Single initial stage loss function.

        Args:
            pts_pred_init (Tensor): Initial point sets prediction with
                shape (N, 9*2)
            bbox_gt_init (Tensor): BBox regression targets in the init stage
                of shape (batch_size, h_i * w_i, 8).
            bbox_weights_init (Tensor): BBox regression loss weights in the
                init stage of shape (batch_size, h_i * w_i, 8).
            stride (int): Point stride.

        Returns:
            tuple:

            - loss_pts_init (Tensor): Initial bbox loss.
            - loss_border_init (Tensor): Initial spatial border loss.
        """
        normalize_term = self.point_base_scale * stride

        bbox_gt_init = bbox_gt_init.reshape(-1, 8)
        bbox_weights_init = bbox_weights_init.reshape(-1)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        pos_ind_init = (bbox_weights_init > 0).nonzero(
            as_tuple=False).reshape(-1)

        pts_pred_init_norm = pts_pred_init[pos_ind_init]
        bbox_gt_init_norm = bbox_gt_init[pos_ind_init]
        bbox_weights_pos_init = bbox_weights_init[pos_ind_init]

        loss_pts_init = self.loss_bbox_init(
            pts_pred_init_norm / normalize_term,
            bbox_gt_init_norm / normalize_term, bbox_weights_pos_init)

        loss_border_init = self.loss_spatial_init(
            pts_pred_init_norm.reshape(-1, 2 * self.num_points) /
            normalize_term,
            bbox_gt_init_norm / normalize_term,
            bbox_weights_pos_init,
            avg_factor=None)

        return loss_pts_init, loss_border_init
