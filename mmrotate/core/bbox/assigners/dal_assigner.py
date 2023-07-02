# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from ..builder import ROTATED_BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator


@ROTATED_BBOX_ASSIGNERS.register_module()
class DalAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox. Each
    proposals will be assigned with `0` or a positive integer indicating the
    ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        scale (float): IoU threshold for positive bboxes.
        pos_num (float): find the nearest pos_num points to gt center in this
        level.
    """

    def __init__(self,
                 angle_version='oc',
                 md_thres=0.5,
                 var=5,
                 num_classes=15,
                 iou_calculator=dict(type='RBboxOverlaps2D')):
        self.angle_version = angle_version
        self.md_thres = md_thres
        self.var = var
        self.num_classes = num_classes
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.gt_max_assign_all = True

    def assign(self,
               bboxes,
               bboxes_pred,
               gt_bboxes,
               epoch_info=(1, 12),
               gt_bboxes_ignore=None,
               gt_labels=None):

        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        # 1. calculate md_overlaps
        sa = self.iou_calculator(bboxes, gt_bboxes)
        assigned_gt_inds = sa.new_full((num_bboxes, ), -1, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = sa.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = sa.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        das = True
        cur_epoch, total_epoch = epoch_info
        alpha = self.calc_mining_param(float(cur_epoch / total_epoch))
        if self.var != -1:
            fa = self.iou_calculator(bboxes_pred, gt_bboxes)

            if self.var == 0:
                md = abs((alpha * sa + (1 - alpha) * fa))
            else:
                md = abs((alpha * sa + (1 - alpha) * fa) -
                         abs(fa - sa)**self.var)
        else:
            das = False
            md = sa

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = md.max(dim=0)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = md.max(dim=1)

        # 2. assign negative: below
        assigned_gt_inds[(max_overlaps >= 0)
                         & (max_overlaps < self.md_thres - 0.1)] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.md_thres
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. match low-quality
        if (gt_max_overlaps < self.md_thres).any():
            below_gt_max_inds = gt_max_overlaps < self.md_thres
            assigned_gt_inds[gt_argmax_overlaps[below_gt_max_inds]] = \
                torch.nonzero(below_gt_max_inds).reshape(-1) + 1
        pos_inds = assigned_gt_inds > 0

        # 5. match weight
        if das:
            pos = md[pos_inds, :]
            max_pos, argmax_pos = pos.max(0)
            comp = (1 - max_pos).expand(pos.size(0), num_gt)
            matching_weight = (comp + pos).gather(
                1, (assigned_gt_inds[pos_inds] - 1).unsqueeze(1)).squeeze(1)

        # 6. assign labels
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        # 7. box_weight and cls_weight
        label_weights = torch.zeros(
            num_bboxes,
            self.num_classes,
            device=assigned_gt_inds.device,
            dtype=torch.float)
        label_weights[assigned_gt_inds >= 0, :] = 1
        label_weights[pos_inds,
                      assigned_labels[pos_inds]] = matching_weight + 1

        box_weights = torch.zeros(
            num_bboxes,
            bboxes.size(1),
            device=assigned_gt_inds.device,
            dtype=torch.float)
        box_weights[pos_inds, :] = matching_weight[:, None]

        # 8. perpare assignresult
        assignresult = AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assignresult.set_extra_property('label_weights', label_weights)
        assignresult.set_extra_property('box_weights', box_weights)

        return assignresult

    def calc_mining_param(self, process, alpha=0.3):
        if process < 0.1:
            bf_weight = 1.0
        elif process > 0.3:
            bf_weight = alpha
        else:
            bf_weight = 5 * (alpha - 1) * process + 1.5 - 0.5 * alpha
        return bf_weight
