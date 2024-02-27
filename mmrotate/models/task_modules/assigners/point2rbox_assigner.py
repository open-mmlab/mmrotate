# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType
from mmengine.structures import InstanceData

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class Point2RBoxAssigner(BaseAssigner):
    """Point2RBoxAssigner between the priors and gt boxes, which can achieve
    balance in positive priors, and gt_bboxes_ignore was not considered for
    now.

    Args:
        pos_ignore_thr (float): the threshold to ignore positive priors
        neg_ignore_thr (float): the threshold to ignore negative priors
        match_times(int): Number of positive priors for each gt box.
           Defaults to 4.
        iou_calculator (:obj:`ConfigDict` or dict): Config dict for iou
            calculator. Defaults to ``dict(type='BboxOverlaps2D')``
    """

    def __init__(
        self,
        pos_ignore_thr: float,
        neg_ignore_thr: float,
        match_times: int = 4,
        iou_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D')
    ) -> None:
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def obb2xyxy(self, obb):
        w = obb[:, 2::5]
        h = obb[:, 3::5]
        a = obb[:, 4::5].detach()
        cosa = torch.cos(a).abs()
        sina = torch.sin(a).abs()
        hbbox_w = cosa * w + sina * h
        hbbox_h = sina * w + cosa * h
        dx = obb[..., 0]
        dy = obb[..., 1]
        dw = hbbox_w.reshape(-1)
        dh = hbbox_h.reshape(-1)
        x1 = dx - dw / 2
        y1 = dy - dh / 2
        x2 = dx + dw / 2
        y2 = dy + dh / 2
        return torch.stack((x1, y1, x2, y2), -1)

    def assign(
            self,
            pred_instances: InstanceData,
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """Assign gt to priors.

        The assignment is done in following steps

        1. assign -1 by default
        2. compute the L1 cost between boxes. Note that we use priors and
           predict boxes both
        3. compute the ignore indexes use gt_bboxes and predict boxes
        4. compute the ignore indexes of positive sample use priors and
           predict boxes


        Args:
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be priors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (:obj:`InstaceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            gt_instances_ignore (:obj:`InstaceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        gt_bboxes = gt_instances.bboxes.tensor
        gt_labels = gt_instances.labels
        gt_bids = gt_instances.bids
        priors = pred_instances.priors
        bbox_pred = pred_instances.decoder_priors
        cls_scores = pred_instances.cls_scores

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              0,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            assign_result = AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
            assign_result.set_extra_property(
                'pos_bbox_mask', bbox_pred.new_empty(0, dtype=torch.bool))
            assign_result.set_extra_property(
                'pos_point_mask', bbox_pred.new_empty(0, dtype=torch.bool))
            assign_result.set_extra_property(
                'pos_predicted_boxes',
                bbox_pred.new_empty((0, bbox_pred.shape[-1])))
            assign_result.set_extra_property(
                'target_boxes', gt_bboxes.new_empty((0, gt_bboxes.shape[-1])))
            assign_result.set_extra_property('target_labels',
                                             gt_labels.new_empty((0, )))
            assign_result.set_extra_property('target_bids',
                                             gt_bids.new_empty((0, )))
            return assign_result

        # 2. Compute the L1 cost between boxes
        # Note that we use priors and predict boxes both
        bbox_pred_xyxy = bbox_pred[:, :4]
        point_mask = gt_bboxes[:, 2] < 1
        gt_bboxes_xyxy = self.obb2xyxy(gt_bboxes)

        cost_center = torch.cdist(
            bbox_xyxy_to_cxcywh(bbox_pred_xyxy)[:, :2], gt_bboxes[:, :2], p=1)
        cost_cls_scores = 1 - cls_scores[:, gt_labels].sigmoid()
        cost_cls_scores[cost_center > 32] = 1e5

        cost_bbox = cost_cls_scores.clone()
        cost_bbox_priors = torch.cdist(
            bbox_xyxy_to_cxcywh(priors),
            bbox_xyxy_to_cxcywh(gt_bboxes_xyxy),
            p=1) * cost_cls_scores
        cost_bbox[:, point_mask] = 1e9
        cost_bbox_priors[:, point_mask] = 1e9

        # 32 is the L1-dist between two adjacent diagonal anchors (stride=16)
        cost_cls_scores[:, ~point_mask] = 1e9

        # We found that topk function has different results in cpu and
        # cuda mode. In order to ensure consistency with the source code,
        # we also use cpu mode.
        # TODO: Check whether the performance of cpu and cuda are the same.
        C = cost_bbox.cpu()
        C1 = cost_bbox_priors.cpu()
        C2 = cost_cls_scores.cpu()

        # self.match_times x n
        index = torch.topk(C, k=self.match_times, dim=0, largest=False)[1]
        index1 = torch.topk(C1, k=self.match_times, dim=0, largest=False)[1]
        index2 = torch.topk(C2, k=self.match_times, dim=0, largest=False)[1]
        strong_idx = index2[:, point_mask.cpu()]

        # (self.match_times*2) x n
        indexes = torch.cat((index, index1, index2),
                            dim=1).reshape(-1).to(bbox_pred.device)

        pred_overlaps = self.iou_calculator(bbox_pred_xyxy, gt_bboxes_xyxy)
        anchor_overlaps = self.iou_calculator(priors, gt_bboxes_xyxy)
        # anchor_overlaps[:, point_mask] = 1
        pred_max_overlaps, _ = pred_overlaps.max(dim=1)
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)

        # 3. Compute the ignore indexes use gt_bboxes and predict boxes
        ignore_idx = pred_max_overlaps > self.neg_ignore_thr
        assigned_gt_inds[ignore_idx] = -1

        # 4. Compute the ignore indexes of positive sample use priors
        # and predict boxes
        pos_gt_index = torch.arange(
            0, C1.size(1),
            device=bbox_pred.device).repeat(self.match_times * 3)
        pos_ious = anchor_overlaps[indexes, pos_gt_index]
        pos_ignore_idx = pos_ious < self.pos_ignore_thr

        # Bbox pos weight, the same as YOLOF
        pos_bbox_idx = ~pos_ignore_idx
        # Point pos weight, False for index and index1, True for index2
        pos_point_idx = pos_ignore_idx.new_zeros(self.match_times, 3,
                                                 C1.size(1))
        pos_point_idx[:, 2, point_mask] = True
        pos_point_idx = pos_point_idx.reshape(-1)
        pos_point_idx = torch.logical_and(pos_point_idx, pos_ious > 0)
        # When the pos is ignored by both bbox and point, not assign gt label
        pos_ignore_idx = torch.logical_and(pos_ignore_idx, ~pos_point_idx)

        pos_gt_index_with_ignore = pos_gt_index + 1
        pos_gt_index_with_ignore[pos_ignore_idx] = -1
        assigned_gt_inds[indexes] = pos_gt_index_with_ignore

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(
            num_gts,
            assigned_gt_inds,
            anchor_max_overlaps,
            labels=assigned_labels)
        assign_result.set_extra_property('pos_point_index', strong_idx)
        assign_result.set_extra_property('pos_bbox_mask', pos_bbox_idx)
        assign_result.set_extra_property('pos_point_mask', pos_point_idx)
        assign_result.set_extra_property('pos_predicted_boxes',
                                         bbox_pred[indexes])
        assign_result.set_extra_property('target_boxes',
                                         gt_bboxes[pos_gt_index])
        assign_result.set_extra_property('target_labels',
                                         gt_labels[pos_gt_index])
        assign_result.set_extra_property('target_bids', gt_bids[pos_gt_index])
        return assign_result
