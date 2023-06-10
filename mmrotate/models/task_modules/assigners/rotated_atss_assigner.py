# Copyright (c) OpenMMLab. All rights reserved.
import torch
import warnings
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.utils import ConfigType
from mmengine.structures import InstanceData
from torch import Tensor
from typing import List, Optional

from mmrotate.registry import TASK_UTILS


def bbox_center_distance(bboxes: Tensor, priors: Tensor) -> Tensor:
    """Compute the center distance between bboxes and priors.

    Args:
        bboxes (Tensor): Shape (n, 5) for , <cx, cy, w, h, t> format.
        priors (Tensor): Shape (n, 5) for priors,
            <cx, cy, w, h, t> format.
    Returns:
        Tensor: Center distances between bboxes and priors.
    """
    bbox_centers = bboxes.centers
    priors_centers = priors.centers
    distances = (priors_centers[:, None, :] -
                 bbox_centers[None, :, :]).pow(2).sum(-1).sqrt()

    return distances


@TASK_UTILS.register_module()
class RotatedATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each prior. Each
    proposals will be assigned with `0` or a positive integer indicating the
    ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    If ``alpha`` is not None, it means that the dynamic cost
    ATSSAssigner is adopted, which is currently only used in the DDOD.
    Args:
        topk (int): number of priors selected in each level
        alpha (float, optional): param of cost rate for each proposal only
            in DDOD. Defaults to None.
        iou_calculator (:obj:`ConfigDict` or dict): Config dict for iou
            calculator. Defaults to ``dict(type='BboxOverlaps2D')``
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes. Defaults to -1.
    """

    def __init__(self,
                 topk: int,
                 alpha: Optional[float] = None,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 ignore_iof_thr: float = -1) -> None:
        self.topk = topk
        self.alpha = alpha
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def assign(
            self,
            pred_instances: InstanceData,
            num_level_priors: List[int],
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """Assign gt to priors.

        The assignment is done in following steps
        1. compute iou between all prior (prior of all pyramid levels) and gt
        2. compute center distance between all prior and gt
        3. on each pyramid level, for each gt, select k prior whose center
           are closest to the gt center, so we total select k*l prior as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        If ``alpha`` is not None, and ``cls_scores`` and `bbox_preds`
        are not None, the overlaps calculation in the first step
        will also include dynamic cost, which is currently only used in
        the DDOD.
        Args:
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            num_level_priors (List): Number of bboxes in each level
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
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        INF = 100000000

        num_gt, num_priors = gt_bboxes.size(0), priors.size(0)

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner,  please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time. '

        # compute iou between all bbox and gt
        if self.alpha is None:
            # ATSSAssigner
            overlaps = self.iou_calculator(priors, gt_bboxes)
            if ('scores' in pred_instances or 'bboxes' in pred_instances):
                warnings.warn(message)

        else:
            # Dynamic cost ATSSAssigner in DDOD
            assert ('scores' in pred_instances
                    and 'bboxes' in pred_instances), message
            cls_scores = pred_instances.scores
            bbox_preds = pred_instances.bboxes

            # compute cls cost for bbox and GT
            cls_cost = torch.sigmoid(cls_scores[:, gt_labels])

            # compute iou between all bbox and gt
            overlaps = self.iou_calculator(bbox_preds, gt_bboxes)

            # make sure that we are in element-wise multiplication
            assert cls_cost.shape == overlaps.shape

            # overlaps is actually a cost matrix
            overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_priors, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_priors == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_priors, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = overlaps.new_full((num_priors, ),
                                                -1,
                                                dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        distances = bbox_center_distance(gt_bboxes, priors)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                priors, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, priors_per_level in enumerate(num_level_priors):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + priors_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, priors_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_priors
        candidate_idxs = candidate_idxs.view(-1)
        priors_cx, priors_cy = priors.centers.unbind(dim=-1)
        ep_priors_cx = priors_cx.view(1, -1).expand(
            num_gt, num_priors).contiguous().view(-1)
        ep_priors_cy = priors_cy.view(1, -1).expand(
            num_gt, num_priors).contiguous().view(-1)
        ep_priors_cx = ep_priors_cx[candidate_idxs].view(-1, num_gt)
        ep_priors_cy = ep_priors_cy[candidate_idxs].view(-1, num_gt)
        ep_priors_centers = torch.stack((ep_priors_cx, ep_priors_cy),
                                        dim=-1).view(-1, 2)
        repeat_gt_bboxes = gt_bboxes.repeat(is_pos.shape[0], 1)
        is_in_gts = repeat_gt_bboxes.find_inside_points(
            ep_priors_centers, is_aligned=True,
            eps=0.01).view(-1, num_gt).to(is_pos.dtype)

        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        assigned_labels = assigned_gt_inds.new_full((num_priors, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
