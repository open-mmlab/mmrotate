# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class ConvexAssigner(BaseAssigner):
    """Assign a corresponding qbbox gt or background to each convex. Each
    proposals will be assigned with `0` or a positive integer indicating the
    ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt


    Args:
        scale (float): qbbox scale for assigning labels. Defaults to 4.0.
        pos_num (int): find the nearest pos_num points to gt center in this
        level. Defaults to 3.
    """

    def __init__(self, scale: float = 4.0, pos_num: int = 3) -> None:
        self.scale = scale
        self.pos_num = pos_num

    def get_horizontal_bboxes(self, gt_rbboxes: Tensor) -> Tensor:
        """get_horizontal_bboxes from polygons.

        Args:
            gt_rbboxes (Tensor): Groundtruth polygons, shape (k, 8).

        Returns:
            gt_rect_bboxes (Tensor): The horizontal bboxes, shape (k, 4).
        """
        gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
        gt_xmin, _ = gt_xs.min(1)
        gt_ymin, _ = gt_ys.min(1)
        gt_xmax, _ = gt_xs.max(1)
        gt_ymax, _ = gt_ys.max(1)
        gt_rect_bboxes = torch.cat([
            gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymax[:,
                                                                          None]
        ],
                                   dim=1)

        return gt_rect_bboxes

    def assign(
            self,
            pred_instances: InstanceData,
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
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
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors are
                convex predicted by the model, shape(n, 18)).
            gt_instances (:obj:`InstaceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes, and the bboxes are GT qboxes, shape (k, 8).
            gt_instances_ignore (:obj:`InstaceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        gt_rbboxes = gt_instances.bboxes
        points = pred_instances.priors
        gt_labels = gt_instances.labels

        num_points = points.shape[0]
        num_gts = gt_rbboxes.shape[0]

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        assert gt_rbboxes.size(1) == 8, 'gt_rbboxes should be (N * 8)'
        gt_bboxes = self.get_horizontal_bboxes(gt_rbboxes)

        # assign gt rbox
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2

        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = points.new_full((num_points, ), float('inf'))
        points_range = torch.arange(points.shape[0], device=points.device)

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            # get the points in this level
            lvl_points = points_xy[lvl_idx, :]
            # get the center point of gt
            gt_point = gt_bboxes_xy[[idx], :]
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]
            # compute the distance between gt center and
            #   all points in this level
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = torch.topk(
                points_gt_dist, self.pos_num, largest=False)
            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]

            # The less_than_recorded_index stores the index
            #   of min_dist that is less then the assigned_gt_dist. Where
            #   assigned_gt_dist stores the dist from previous assigned gt
            #   (if exist) to each point.
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_points_index]
            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[
                less_than_recorded_index]
            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[
                less_than_recorded_index]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points, ),
                                                        -1,
                                                        dtype=torch.long)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
