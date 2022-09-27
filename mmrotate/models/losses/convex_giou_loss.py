# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.ops import convex_giou
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from mmrotate.registry import MODELS


class ConvexGIoULossFuction(Function):
    """The function of Convex GIoU loss."""

    @staticmethod
    def forward(ctx,
                pred,
                target,
                weight=None,
                reduction=None,
                avg_factor=None,
                loss_weight=1.0):
        """Forward function.

        Args:
            ctx:  {save_for_backward, convex_points_grad}
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): The reduction method of the
            loss. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            loss_weight (float, optional): The weight of loss. Defaults to 1.0.
        """
        ctx.save_for_backward(pred)
        convex_gious, grad = convex_giou(pred, target)

        loss = 1 - convex_gious
        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()

        unvaild_inds = torch.nonzero((grad > 1).sum(1), as_tuple=False)[:, 0]
        grad[unvaild_inds] = 1e-6

        # _reduce_grad
        reduce_grad = -grad / grad.size(0) * loss_weight
        ctx.convex_points_grad = reduce_grad
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, input=None):
        """Backward function."""
        convex_points_grad = ctx.convex_points_grad
        return convex_points_grad, None, None, None, None, None


convex_giou_loss = ConvexGIoULossFuction.apply


@MODELS.register_module()
class ConvexGIoULoss(nn.Module):
    """Convex GIoU loss.

    Computing the Convex GIoU loss between a set of predicted convexes and
    target convexes.
    Args:
        reduction (str, optional): The reduction method of the loss. Defaults
            to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    Return:
        torch.Tensor: Loss tensor.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ConvexGIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * convex_giou_loss(
            pred, target, weight, reduction, avg_factor, self.loss_weight)
        return loss


class BCConvexGIoULossFuction(Function):
    """The function of BCConvex GIoU loss."""

    @staticmethod
    def forward(ctx,
                pred,
                target,
                weight=None,
                reduction=None,
                avg_factor=None,
                loss_weight=1.0):
        """Forward function.

        Args:
            ctx:  {save_for_backward, convex_points_grad}
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): The reduction method of the
            loss. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            loss_weight (float, optional): The weight of loss. Defaults to 1.0.
        """
        ctx.save_for_backward(pred)
        convex_gious, grad = convex_giou(pred, target)

        pts_pred_all_dx = pred[:, 0::2]
        pts_pred_all_dy = pred[:, 1::2]

        pred_left_x_inds = pts_pred_all_dx.min(dim=1, keepdim=True)[1]
        pred_right_x_inds = pts_pred_all_dx.max(dim=1, keepdim=True)[1]
        pred_up_y_inds = pts_pred_all_dy.min(dim=1, keepdim=True)[1]
        pred_bottom_y_inds = pts_pred_all_dy.max(dim=1, keepdim=True)[1]

        pred_right_x = pts_pred_all_dx.gather(dim=1, index=pred_right_x_inds)
        pred_right_y = pts_pred_all_dy.gather(dim=1, index=pred_right_x_inds)

        pred_left_x = pts_pred_all_dx.gather(dim=1, index=pred_left_x_inds)
        pred_left_y = pts_pred_all_dy.gather(dim=1, index=pred_left_x_inds)

        pred_up_x = pts_pred_all_dx.gather(dim=1, index=pred_up_y_inds)
        pred_up_y = pts_pred_all_dy.gather(dim=1, index=pred_up_y_inds)

        pred_bottom_x = pts_pred_all_dx.gather(dim=1, index=pred_bottom_y_inds)
        pred_bottom_y = pts_pred_all_dy.gather(dim=1, index=pred_bottom_y_inds)
        pred_corners = torch.cat([
            pred_left_x, pred_left_y, pred_up_x, pred_up_y, pred_right_x,
            pred_right_y, pred_bottom_x, pred_bottom_y
        ],
                                 dim=-1)

        pts_target_all_dx = target[:, 0::2]
        pts_target_all_dy = target[:, 1::2]

        target_left_x_inds = pts_target_all_dx.min(dim=1, keepdim=True)[1]
        target_right_x_inds = pts_target_all_dx.max(dim=1, keepdim=True)[1]
        target_up_y_inds = pts_target_all_dy.min(dim=1, keepdim=True)[1]
        target_bottom_y_inds = pts_target_all_dy.max(dim=1, keepdim=True)[1]

        target_right_x = pts_target_all_dx.gather(
            dim=1, index=target_right_x_inds)
        target_right_y = pts_target_all_dy.gather(
            dim=1, index=target_right_x_inds)

        target_left_x = pts_target_all_dx.gather(
            dim=1, index=target_left_x_inds)
        target_left_y = pts_target_all_dy.gather(
            dim=1, index=target_left_x_inds)

        target_up_x = pts_target_all_dx.gather(dim=1, index=target_up_y_inds)
        target_up_y = pts_target_all_dy.gather(dim=1, index=target_up_y_inds)

        target_bottom_x = pts_target_all_dx.gather(
            dim=1, index=target_bottom_y_inds)
        target_bottom_y = pts_target_all_dy.gather(
            dim=1, index=target_bottom_y_inds)

        target_corners = torch.cat([
            target_left_x, target_left_y, target_up_x, target_up_y,
            target_right_x, target_right_y, target_bottom_x, target_bottom_y
        ],
                                   dim=-1)

        pts_pred_dx_mean = pts_pred_all_dx.mean(
            dim=1, keepdim=True).reshape(-1, 1)
        pts_pred_dy_mean = pts_pred_all_dy.mean(
            dim=1, keepdim=True).reshape(-1, 1)
        pts_pred_mean = torch.cat([pts_pred_dx_mean, pts_pred_dy_mean], dim=-1)

        pts_target_dx_mean = pts_target_all_dx.mean(
            dim=1, keepdim=True).reshape(-1, 1)
        pts_target_dy_mean = pts_target_all_dy.mean(
            dim=1, keepdim=True).reshape(-1, 1)
        pts_target_mean = torch.cat([pts_target_dx_mean, pts_target_dy_mean],
                                    dim=-1)

        beta = 1.0

        diff_mean = torch.abs(pts_pred_mean - pts_target_mean)
        diff_mean_loss = torch.where(diff_mean < beta,
                                     0.5 * diff_mean * diff_mean / beta,
                                     diff_mean - 0.5 * beta)
        diff_mean_loss = diff_mean_loss.sum() / len(diff_mean_loss)

        diff_corners = torch.abs(pred_corners - target_corners)
        diff_corners_loss = torch.where(
            diff_corners < beta, 0.5 * diff_corners * diff_corners / beta,
            diff_corners - 0.5 * beta)
        diff_corners_loss = diff_corners_loss.sum() / len(diff_corners_loss)

        target_aspect = AspectRatio(target)
        smooth_loss_weight = torch.exp((-1 / 4) * target_aspect)
        loss = \
            smooth_loss_weight * (diff_mean_loss.reshape(-1, 1).cuda() +
                                  diff_corners_loss.reshape(-1, 1).cuda()) + \
            1 - (1 - 2 * smooth_loss_weight) * convex_gious

        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()

        unvaild_inds = torch.nonzero((grad > 1).sum(1), as_tuple=False)[:, 0]
        grad[unvaild_inds] = 1e-6

        reduce_grad = -grad / grad.size(0) * loss_weight
        ctx.convex_points_grad = reduce_grad
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, input=None):
        """Backward function."""
        convex_points_grad = ctx.convex_points_grad
        return convex_points_grad, None, None, None, None, None


bc_convex_giou_loss = BCConvexGIoULossFuction.apply


@MODELS.register_module()
class BCConvexGIoULoss(nn.Module):
    """BCConvex GIoU loss.

    Computing the BCConvex GIoU loss between a set of predicted convexes and
    target convexes.
    Args:
        reduction (str, optional): The reduction method of the loss. Defaults
            to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    Return:
        torch.Tensor: Loss tensor.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(BCConvexGIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * bc_convex_giou_loss(
            pred, target, weight, reduction, avg_factor, self.loss_weight)
        return loss


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
