# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmdet.models.utils import images_to_levels, multi_apply, unmap
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.models.dense_heads.rotated_reppoints_head import \
    RotatedRepPointsHead
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import qbox2rbox
from ..utils import get_num_level_anchors_inside, points_center_pts


@MODELS.register_module()
class SAMRepPointsHead(RotatedRepPointsHead):
    """SAM RepPoints head."""

    def _get_targets_single(self,
                            flat_proposals: Tensor,
                            valid_flags: Tensor,
                            num_level_proposals: List[int],
                            gt_instances: InstanceData,
                            gt_instances_ignore: InstanceData,
                            stage: str = 'init',
                            unmap_outputs: bool = True) -> tuple:
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            flat_proposals (Tensor): Multi level points of a image.
            valid_flags (Tensor): Multi level valid flags of a image.
            num_level_proposals (List[int]): Number of anchors of each scale
                level.
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
        num_level_proposals_inside = get_num_level_anchors_inside(
            num_level_proposals, inside_flags)
        pred_instances = InstanceData(priors=proposals)

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
            assign_result = assigner.assign(pred_instances, gt_instances,
                                            gt_instances_ignore)
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight

            if self.train_cfg.refine.assigner['type'] not in (
                    'ATSSAssigner', 'ATSSConvexAssigner', 'SASAssigner'):
                assign_result = assigner.assign(pred_instances, gt_instances,
                                                gt_instances_ignore)
            else:
                assign_result = assigner.assign(pred_instances,
                                                num_level_proposals_inside,
                                                gt_instances,
                                                gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

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

        # use la
        rbboxes_center, width, height, angles = torch.split(
            qbox2rbox(bbox_gt), [2, 1, 1, 1], dim=-1)

        if stage == 'init':
            points_xy = pos_proposals[:, :2]
        else:
            points_xy = points_center_pts(pos_proposals, y_first=True)

        distances = torch.zeros_like(angles).reshape(-1)

        angles_index_wh = ((width != 0) & (angles >= 0) &
                           (angles <= 1.57)).squeeze()
        angles_index_hw = ((width != 0) & ((angles < 0) |
                                           (angles > 1.57))).squeeze()

        # 01_la:compution of distance
        distances[angles_index_wh] = torch.sqrt(
            (torch.pow(
                rbboxes_center[angles_index_wh, 0] -
                points_xy[angles_index_wh, 0], 2) /
             width[angles_index_wh].squeeze()) +
            (torch.pow(
                rbboxes_center[angles_index_wh, 1] -
                points_xy[angles_index_wh, 1], 2) /
             height[angles_index_wh].squeeze()))

        distances[angles_index_hw] = torch.sqrt(
            (torch.pow(
                rbboxes_center[angles_index_hw, 0] -
                points_xy[angles_index_hw, 0], 2) /
             height[angles_index_hw].squeeze()) +
            (torch.pow(
                rbboxes_center[angles_index_hw, 1] -
                points_xy[angles_index_hw, 1], 2) /
             width[angles_index_hw].squeeze()))
        distances[distances == float('nan')] = 0.

        sam_weights = label_weights * (torch.exp(1 / (distances + 1)))
        sam_weights[sam_weights == float('inf')] = 0.

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

            sam_weights = unmap(sam_weights, num_total_proposals, inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, sampling_result,
                sam_weights)

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
                refine stage. Defaults to 'init'.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

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
        num_level_proposals_list = [num_level_proposals] * num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_results_list, all_sam_weights) = multi_apply(
             self._get_targets_single,
             proposals_list,
             valid_flag_list,
             num_level_proposals_list,
             batch_gt_instances,
             batch_gt_instances_ignore,
             stage=stage,
             unmap_outputs=unmap_outputs)

        # sampled points of all images
        avg_refactor = sum(
            [results.avg_factor for results in sampling_results_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)
        sam_weights_list = images_to_levels(all_sam_weights,
                                            num_level_proposals)
        res = (labels_list, label_weights_list, bbox_gt_list, proposals_list,
               proposal_weights_list, avg_refactor, sam_weights_list)

        return res

    def loss_by_feat_single(self, cls_score: Tensor, pts_pred_init: Tensor,
                            pts_pred_refine: Tensor, labels: Tensor,
                            label_weights, bbox_gt_init: Tensor,
                            bbox_weights_init: Tensor,
                            sam_weights_init: Tensor, bbox_gt_refine: Tensor,
                            bbox_weights_refine: Tensor,
                            sam_weights_refine: Tensor, stride: int,
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
            sam_weights_init (Tensor):
            bbox_gt_refine (Tensor): BBox regression targets in the refine
                stage of shape (batch_size, h_i * w_i, 8).
            bbox_weights_refine (Tensor): BBox regression loss weights in the
                refine stage of shape (batch_size, h_i * w_i, 8).
            sam_weights_refine (Tensor):
            stride (int): Point stride.
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
        sam_weights_init = sam_weights_init.reshape(-1)
        pos_ind_init = (bbox_weights_init > 0).nonzero(
            as_tuple=False).reshape(-1)
        pos_bbox_gt_init = bbox_gt_init[pos_ind_init]
        pos_pts_pred_init = pts_pred_init[pos_ind_init]
        pos_bbox_weights_init = bbox_weights_init[pos_ind_init]
        sam_weights_pos_init = sam_weights_init[pos_ind_init]
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(
            pos_pts_pred_init / normalize_term,
            pos_bbox_gt_init / normalize_term,
            pos_bbox_weights_init * sam_weights_pos_init)

        # refine loss
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 8)
        pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        bbox_weights_refine = bbox_weights_refine.reshape(-1)
        sam_weights_refine = sam_weights_refine.reshape(-1)
        pos_ind_refine = (bbox_weights_refine > 0).nonzero(
            as_tuple=False).reshape(-1)
        pos_bbox_gt_refine = bbox_gt_refine[pos_ind_refine]
        pos_pts_pred_refine = pts_pred_refine[pos_ind_refine]
        pos_bbox_weights_refine = bbox_weights_refine[pos_ind_refine]
        sam_weights_pos_refine = sam_weights_refine[pos_ind_refine]
        loss_pts_refine = self.loss_bbox_refine(
            pos_pts_pred_refine / normalize_term,
            pos_bbox_gt_refine / normalize_term,
            pos_bbox_weights_refine * sam_weights_pos_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        pts_preds_init: List[Tensor],
        pts_preds_refine: List[Tensor],
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
         avg_factor_init, sam_weights_list_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       batch_img_metas, device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
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
         candidate_list_refine, bbox_weights_list_refine, avg_factor_refine,
         sam_weights_list_refine) = cls_reg_targets_refine

        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            sam_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            sam_weights_list_refine,
            self.point_strides,
            avg_factor_refine=avg_factor_refine)
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        return loss_dict_all
