# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.ops import points_in_polygons

from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class SpatialBorderLoss(nn.Module):
    """Spatial Border loss for learning points in Oriented RepPoints.

    Args:
        pts (torch.Tensor): point sets with shape (N, 9*2).
        Default points number in each point set is 9.
        gt_bboxes (torch.Tensor): gt_bboxes with polygon form with shape(N, 8)

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self, loss_weight=1.0):
        super(SpatialBorderLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pts, gt_bboxes, weight, *args, **kwargs):
        loss = self.loss_weight * weighted_spatial_border_loss(
            pts, gt_bboxes, weight, *args, **kwargs)
        return loss


def spatial_border_loss(pts, gt_bboxes):
    """The loss is used to penalize the learning points out of the assigned
    ground truth boxes (polygon by default).

    Args:
        pts (torch.Tensor): point sets with shape (N, 9*2).
        gt_bboxes (torch.Tensor): gt_bboxes with polygon form with shape(N, 8)

    Returns:
        loss (torch.Tensor)
    """
    num_gts, num_pointsets = gt_bboxes.size(0), pts.size(0)
    num_point = int(pts.size(1) / 2.0)
    loss = pts.new_zeros([0])

    if num_gts > 0:
        inside_flag_list = []
        for i in range(num_point):
            pt = pts[:, (2 * i):(2 * i + 2)].reshape(num_pointsets,
                                                     2).contiguous()
            inside_pt_flag = points_in_polygons(pt, gt_bboxes)
            inside_pt_flag = torch.diag(inside_pt_flag)
            inside_flag_list.append(inside_pt_flag)

        inside_flag = torch.stack(inside_flag_list, dim=1)
        pts = pts.reshape(-1, num_point, 2)
        out_border_pts = pts[torch.where(inside_flag == 0)]

        if out_border_pts.size(0) > 0:
            corr_gt_boxes = gt_bboxes[torch.where(inside_flag == 0)[0]]
            corr_gt_boxes_center_x = (corr_gt_boxes[:, 0] +
                                      corr_gt_boxes[:, 4]) / 2.0
            corr_gt_boxes_center_y = (corr_gt_boxes[:, 1] +
                                      corr_gt_boxes[:, 5]) / 2.0
            corr_gt_boxes_center = torch.stack(
                [corr_gt_boxes_center_x, corr_gt_boxes_center_y], dim=1)
            distance_out_pts = 0.2 * ((
                (out_border_pts - corr_gt_boxes_center)**2).sum(dim=1).sqrt())
            loss = distance_out_pts.sum() / out_border_pts.size(0)

    return loss


def weighted_spatial_border_loss(pts, gt_bboxes, weight, avg_factor=None):
    """Weghted spatial border loss.

    Args:
        pts (torch.Tensor): point sets with shape (N, 9*2).
        gt_bboxes (torch.Tensor): gt_bboxes with polygon form with shape(N, 8)
        weight (torch.Tensor): weights for point sets with shape (N)

    Returns:
        loss (torch.Tensor)
    """

    weight = weight.unsqueeze(dim=1).repeat(1, 4)
    assert weight.dim() == 2
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = spatial_border_loss(pts, gt_bboxes)

    return torch.sum(loss)[None] / avg_factor
