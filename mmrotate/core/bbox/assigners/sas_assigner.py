# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import convex_iou, points_in_polygons
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from ..builder import ROTATED_BBOX_ASSIGNERS


def convex_overlaps(gt_rbboxes, points):
    """Compute overlaps between polygons and points.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).
        points (torch.Tensor): Points to be assigned, shape(n, 18).

    Returns:
        overlaps (torch.Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
    """
    if gt_rbboxes.shape[0] == 0:
        return gt_rbboxes.new_zeros((0, points.shape[0]))
    overlaps = convex_iou(points, gt_rbboxes)
    return overlaps


def get_horizontal_bboxes(gt_rbboxes):
    """Get horizontal bboxes from polygons.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).

    Returns:
        gt_rect_bboxes (torch.Tensor): The horizontal bboxes, shape (k, 4).
    """
    gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
    gt_xmin, _ = gt_xs.min(1)
    gt_ymin, _ = gt_ys.min(1)
    gt_xmax, _ = gt_xs.max(1)
    gt_ymax, _ = gt_ys.max(1)
    gt_rect_bboxes = torch.cat([
        gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymax[:, None]
    ],
                               dim=1)
    return gt_rect_bboxes


def AspectRatio(gt_rbboxes):
    """Compute the aspect ratio of all gts.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).

    Returns:
        ratios (torch.Tensor): The aspect ratio of gt_rbboxes, shape (k, 1).
    """
    pt1, pt2, pt3, pt4 = gt_rbboxes[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    ratios = (width / height)
    return ratios


@ROTATED_BBOX_ASSIGNERS.register_module()
class SASAssigner(BaseAssigner):
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

    def __init__(self, topk):
        self.topk = topk

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        Args:
            bboxes (torch.Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (torch.Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000

        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        overlaps = convex_overlaps(gt_bboxes, bboxes)
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of poly
        gt_bboxes_hbb = get_horizontal_bboxes(gt_bboxes)

        gt_cx = (gt_bboxes_hbb[:, 0] + gt_bboxes_hbb[:, 2]) / 2.0
        gt_cy = (gt_bboxes_hbb[:, 1] + gt_bboxes_hbb[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes = bboxes.reshape(-1, 9, 2)
        pts_x = bboxes[:, :, 0::2]
        pts_y = bboxes[:, :, 1::2]

        pts_x_mean = pts_x.mean(dim=1).squeeze()
        pts_y_mean = pts_y.mean(dim=1).squeeze()

        bboxes_points = torch.stack((pts_x_mean, pts_y_mean), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        gt_bboxes_ratios = AspectRatio(gt_bboxes)
        gt_bboxes_ratios_per_gt = gt_bboxes_ratios.mean(0)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        # new assign
        iou_thr_weight = torch.exp((-1 / 4) * gt_bboxes_ratios_per_gt)
        overlaps_thr_per_gt = overlaps_thr_per_gt * iou_thr_weight
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        # inside_flag = torch.full([num_bboxes, num_gt],
        #                          0.).to(gt_bboxes.device).float()
        inside_flag = points_in_polygons(bboxes_points, gt_bboxes)
        # pointsJf(bboxes_points, gt_bboxes, inside_flag)
        is_in_gts = inside_flag[candidate_idxs,
                                torch.arange(num_gt)].to(is_pos.dtype)

        is_pos = is_pos & is_in_gts
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

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

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
