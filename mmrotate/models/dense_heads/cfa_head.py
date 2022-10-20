# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmdet.models.utils import multi_apply
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.models.dense_heads.rotated_reppoints_head import \
    RotatedRepPointsHead
from mmrotate.registry import MODELS
from ..utils import convex_overlaps, levels_to_images


@MODELS.register_module()
class CFAHead(RotatedRepPointsHead):
    """CFA head.

    Args:
        topk (int): Number of the highest topk points. Defaults to 6.
        anti_factor (float): Feature anti-aliasing coefficient.
            Defaults to 0.75.
    """  # noqa: W605

    def __init__(self,
                 *args,
                 topk: int = 6,
                 anti_factor: float = 0.75,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.anti_factor = anti_factor

    def loss_by_feat_single(self, pts_pred_init: Tensor, bbox_gt_init: Tensor,
                            bbox_weights_init: Tensor, stride: int,
                            avg_factor_init: int) -> Tuple[Tensor]:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            pts_pred_init (Tensor): Points of shape
                (batch_size, h_i * w_i, num_points * 2).
            bbox_gt_init (Tensor): BBox regression targets in the init stage
                of shape (batch_size, h_i * w_i, 8).
            bbox_weights_init (Tensor): BBox regression loss weights in the
                init stage of shape (batch_size, h_i * w_i, 8).
            stride (int): Point stride.
            avg_factor_init (int): Average factor that is used to average
                the loss in the init stage.

        Returns:
            Tuple[Tensor]: loss components.
        """
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

        return loss_pts_init,

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
            stage='init',
            return_sampling_results=False)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         avg_factor_init) = cls_reg_targets_init

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

        cls_reg_targets_refine = self.get_cfa_targets(
            bbox_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='refine',
            return_sampling_results=False)
        (labels_list, label_weights_list, bbox_gt_list_refine, _,
         bbox_weights_list_refine, pos_inds_list_refine,
         pos_gt_index_list_refine) = cls_reg_targets_refine
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        pts_coordinate_preds_init_cfa = levels_to_images(
            pts_coordinate_preds_init, flatten=True)
        pts_coordinate_preds_init_cfa = [
            item.reshape(-1, 2 * self.num_points)
            for item in pts_coordinate_preds_init_cfa
        ]
        pts_coordinate_preds_refine = levels_to_images(
            pts_coordinate_preds_refine, flatten=True)
        pts_coordinate_preds_refine = [
            item.reshape(-1, 2 * self.num_points)
            for item in pts_coordinate_preds_refine
        ]

        with torch.no_grad():
            pos_losses_list, = multi_apply(self.get_pos_loss, cls_scores,
                                           pts_coordinate_preds_init_cfa,
                                           labels_list, bbox_gt_list_refine,
                                           label_weights_list,
                                           bbox_weights_list_refine,
                                           pos_inds_list_refine)
            labels_list, label_weights_list, bbox_weights_list_refine, \
                num_pos, pos_normalize_term = multi_apply(
                    self.reassign,
                    pos_losses_list,
                    labels_list,
                    label_weights_list,
                    pts_coordinate_preds_init_cfa,
                    bbox_weights_list_refine,
                    batch_gt_instances,
                    pos_inds_list_refine,
                    pos_gt_index_list_refine,
                    num_proposals_each_level=num_proposals_each_level,
                    num_level=num_level
                )
            num_pos = sum(num_pos)
        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        pts_preds_refine = torch.cat(pts_coordinate_preds_refine, 0).view(
            -1, pts_coordinate_preds_refine[0].size(-1))
        labels = torch.cat(labels_list, 0).view(-1)
        labels_weight = torch.cat(label_weights_list, 0).view(-1)
        rbbox_gt_refine = torch.cat(bbox_gt_list_refine,
                                    0).view(-1,
                                            bbox_gt_list_refine[0].size(-1))
        convex_weights_refine = torch.cat(bbox_weights_list_refine, 0).view(-1)
        pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)
        pos_inds_flatten = ((0 <= labels) &
                            (labels < self.num_classes)).nonzero(
                                as_tuple=False).reshape(-1)
        assert len(pos_normalize_term) == len(pos_inds_flatten)
        if num_pos:
            losses_cls = self.loss_cls(
                cls_scores, labels, labels_weight, avg_factor=num_pos)
            pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
            pos_rbbox_gt_refine = rbbox_gt_refine[pos_inds_flatten]
            pos_convex_weights_refine = convex_weights_refine[pos_inds_flatten]
            losses_pts_refine = self.loss_bbox_refine(
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_convex_weights_refine)
        else:
            losses_cls = cls_scores.sum() * 0
            losses_pts_refine = pts_preds_refine.sum() * 0

        losses_pts_init, = multi_apply(
            self.loss_by_feat_single,
            pts_coordinate_preds_init,
            bbox_gt_list_init,
            bbox_weights_list_init,
            self.point_strides,
            avg_factor_init=avg_factor_init,
        )
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        return loss_dict_all

    def get_cfa_targets(self,
                        proposals_list: List[Tensor],
                        valid_flag_list: List[Tensor],
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        batch_gt_instances_ignore: OptInstanceList = None,
                        stage: str = 'init',
                        unmap_outputs: bool = True,
                        return_sampling_results: bool = False) -> tuple:
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
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple:

            - all_labels (list[Tensor]): Labels of each level.
            - all_label_weights (list[Tensor]): Label weights of each
            level.
            - all_bbox_gt (list[Tensor]): Ground truth bbox of each level.
            - all_proposals (list[Tensor]): Proposals(points/bboxes) of
            each level.
            - all_proposal_weights (list[Tensor]): Proposal weights of
            each level.
            - pos_inds (list[Tensor]): Index of positive samples in all
            images.
            - gt_inds (list[Tensor]): Index of ground truth bbox in all
            images.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(batch_img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_result) = multi_apply(
             self._get_targets_single,
             proposals_list,
             valid_flag_list,
             batch_gt_instances,
             batch_gt_instances_ignore,
             stage=stage,
             unmap_outputs=unmap_outputs)
        pos_inds = []
        for i, single_labels in enumerate(all_labels):
            pos_mask = (0 <= single_labels) & (
                single_labels < self.num_classes)
            pos_inds.append(pos_mask.nonzero(as_tuple=False).view(-1))

        gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]

        return (all_labels, all_label_weights, all_bbox_gt, all_proposals,
                all_proposal_weights, pos_inds, gt_inds)

    def get_pos_loss(self, cls_score: Tensor, pts_pred: Tensor, label: Tensor,
                     bbox_gt: Tensor, label_weight: Tensor,
                     convex_weight: Tensor, pos_inds: Tensor) -> Tensor:
        """Calculate loss of all potential positive samples obtained from first
        match process.

        Args:
            cls_score (Tensor): Box scores of single image with shape
                (num_anchors, num_classes)
            pts_pred (Tensor): Box energies / deltas of single image
                with shape (num_anchors, 4)
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            bbox_gt (Tensor): Ground truth box.
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            convex_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.

        Returns:
            Tensor: Losses of all positive samples in single image.
        """
        # avoid no positive samplers
        if pos_inds.shape[0] == 0:
            pos_scores = cls_score
            pos_pts_pred = pts_pred
            pos_bbox_gt = bbox_gt
            pos_label = label
            pos_label_weight = label_weight
            pos_convex_weight = convex_weight
        else:
            pos_scores = cls_score[pos_inds]
            pos_pts_pred = pts_pred[pos_inds]
            pos_bbox_gt = bbox_gt[pos_inds]
            pos_label = label[pos_inds]
            pos_label_weight = label_weight[pos_inds]
            pos_convex_weight = convex_weight[pos_inds]
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        loss_bbox = self.loss_bbox_refine(
            pos_pts_pred,
            pos_bbox_gt,
            pos_convex_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        return pos_loss,

    def reassign(self,
                 pos_losses: Tensor,
                 label: Tensor,
                 label_weight: Tensor,
                 pts_pred_init: Tensor,
                 convex_weight: Tensor,
                 gt_instances: InstanceData,
                 pos_inds: Tensor,
                 pos_gt_inds: Tensor,
                 num_proposals_each_level: Optional[List] = None,
                 num_level: Optional[int] = None) -> tuple:
        """CFA reassign process.

        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            pts_pred_init (Tensor):
            convex_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            num_proposals_each_level (list, optional): Number of proposals
                of each level.
            num_level (int, optional): Number of level.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

            - label (Tensor): classification target of each anchor after
            paa assign, with shape (num_anchors,)
            - label_weight (Tensor): Classification loss weight of each
            anchor after paa assign, with shape (num_anchors).
            - convex_weight (Tensor): Bbox weight of each anchor with
            shape (num_anchors, 4).
            - pos_normalize_term (list): pos normalize term for refine
            points losses.
        """
        if len(pos_inds) == 0:
            return label, label_weight, convex_weight, 0, torch.tensor(
                []).type_as(convex_weight)

        num_gt = pos_gt_inds.max() + 1
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)

        overlaps_matrix = convex_overlaps(gt_instances['bboxes'],
                                          pts_pred_init)
        pos_inds_after_cfa = []
        ignore_inds_after_cfa = []
        re_assign_weights_after_cfa = []
        for gt_ind in range(num_gt):
            pos_inds_cfa = []
            pos_loss_cfa = []
            pos_overlaps_init_cfa = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_cfa.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_cfa.append(value)
                pos_overlaps_init_cfa.append(
                    overlaps_matrix[:, pos_inds[level_gt_mask][topk_inds]])
            pos_inds_cfa = torch.cat(pos_inds_cfa)
            pos_loss_cfa = torch.cat(pos_loss_cfa)
            pos_overlaps_init_cfa = torch.cat(pos_overlaps_init_cfa, 1)
            if len(pos_inds_cfa) < 2:
                pos_inds_after_cfa.append(pos_inds_cfa)
                ignore_inds_after_cfa.append(pos_inds_cfa.new_tensor([]))
                re_assign_weights_after_cfa.append(
                    pos_loss_cfa.new_ones([len(pos_inds_cfa)]))
            else:
                pos_loss_cfa, sort_inds = pos_loss_cfa.sort()
                pos_inds_cfa = pos_inds_cfa[sort_inds]
                pos_overlaps_init_cfa = pos_overlaps_init_cfa[:, sort_inds] \
                    .reshape(-1, len(pos_inds_cfa))
                pos_loss_cfa = pos_loss_cfa.reshape(-1)
                loss_mean = pos_loss_cfa.mean()
                loss_var = pos_loss_cfa.var()

                gauss_prob_density = \
                    (-(pos_loss_cfa - loss_mean) ** 2 / loss_var) \
                    .exp() / loss_var.sqrt()
                index_inverted, _ = torch.arange(
                    len(gauss_prob_density)).sort(descending=True)
                gauss_prob_inverted = torch.cumsum(
                    gauss_prob_density[index_inverted], 0)
                gauss_prob = gauss_prob_inverted[index_inverted]
                gauss_prob_norm = (gauss_prob - gauss_prob.min()) / \
                                  (gauss_prob.max() - gauss_prob.min())

                # splitting by gradient consistency
                loss_curve = gauss_prob_norm * pos_loss_cfa
                _, max_thr = loss_curve.topk(1)
                reweights = gauss_prob_norm[:max_thr + 1]
                # feature anti-aliasing coefficient
                pos_overlaps_init_cfa = pos_overlaps_init_cfa[:, :max_thr + 1]
                overlaps_level = pos_overlaps_init_cfa[gt_ind] / (
                    pos_overlaps_init_cfa.sum(0) + 1e-6)
                reweights = \
                    self.anti_factor * overlaps_level * reweights + \
                    1e-6
                re_assign_weights = \
                    reweights.reshape(-1) / reweights.sum() * \
                    torch.ones(len(reweights)).type_as(
                        gauss_prob_norm).sum()
                pos_inds_temp = pos_inds_cfa[:max_thr + 1]
                ignore_inds_temp = pos_inds_cfa.new_tensor([])

                pos_inds_after_cfa.append(pos_inds_temp)
                ignore_inds_after_cfa.append(ignore_inds_temp)
                re_assign_weights_after_cfa.append(re_assign_weights)

        pos_inds_after_cfa = torch.cat(pos_inds_after_cfa)
        ignore_inds_after_cfa = torch.cat(ignore_inds_after_cfa)
        re_assign_weights_after_cfa = torch.cat(re_assign_weights_after_cfa)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_cfa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_cfa] = 0
        convex_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_cfa)

        re_assign_weights_mask = (
            pos_inds.unsqueeze(1) == pos_inds_after_cfa).any(1)
        reweight_ids = pos_inds[re_assign_weights_mask]
        label_weight[reweight_ids] = re_assign_weights_after_cfa
        convex_weight[reweight_ids] = re_assign_weights_after_cfa

        pos_level_mask_after_cfa = []
        for i in range(num_level):
            mask = (pos_inds_after_cfa >= inds_level_interval[i]) & (
                pos_inds_after_cfa < inds_level_interval[i + 1])
            pos_level_mask_after_cfa.append(mask)
        pos_level_mask_after_cfa = torch.stack(pos_level_mask_after_cfa,
                                               0).type_as(label)
        pos_normalize_term = pos_level_mask_after_cfa * (
            self.point_base_scale *
            torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[
            pos_normalize_term > 0].type_as(convex_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_cfa)

        return label, label_weight, convex_weight, num_pos, pos_normalize_term
