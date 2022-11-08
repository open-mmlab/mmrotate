# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from mmrotate.registry import MODELS
from mmrotate.structures import norm_angle


@weighted_loss
def rd_iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')

    pred_x, pred_y, pred_w, pred_h, pred_t = pred.split([1, 1, 1, 1, 1],
                                                        dim=-1)
    target_x, target_y, target_w, target_h, target_t = target.split(
        [1, 1, 1, 1, 1], dim=-1)

    target_z = torch.zeros_like(target_t)
    target_l = torch.ones_like(target_t) * 0.5 * np.pi

    pred_z = torch.ones_like(pred_t) * norm_angle(pred_t - target_t, 'le90')
    pred_l = torch.ones_like(pred_t) * 0.5 * np.pi

    area_pred = pred_w * pred_h * pred_l
    area_target = target_w * target_h * target_l

    union = (
        f(pred_x, target_x, pred_w, target_w) *
        f(pred_y, target_y, pred_h, target_h) *
        f(pred_z, target_z, pred_l, target_l))

    ious = union / (area_pred + area_target - union)

    enclose_area = (
        f2(pred_x, target_x, pred_w, target_w) *
        f2(pred_y, target_y, pred_h, target_h) *
        f2(pred_z, target_z, pred_l, target_l))

    gious = ious - (enclose_area - union) / enclose_area

    # ious = ious.squeeze(0).clamp(min=eps)

    loss = 1 - gious.squeeze(-1)

    return loss


def f(x1, x2, w1, w2):
    ff = torch.min(x1 + 0.5 * w1, x2 + 0.5 * w2) - torch.max(
        x1 - 0.5 * w1, x2 - 0.5 * w2)
    return ff.clamp(min=0)


def f2(x1, x2, w1, w2):
    ff = torch.max(x1 + 0.5 * w1, x2 + 0.5 * w2) - torch.min(
        x1 - 0.5 * w1, x2 - 0.5 * w2)
    return ff.clamp(min=0)


@MODELS.register_module()
class RDIoULoss(nn.Module):
    """RotatedIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(RDIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
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
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rd_iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
