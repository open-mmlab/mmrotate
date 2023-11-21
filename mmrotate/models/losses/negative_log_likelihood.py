# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss
from mmdet.models.losses import smooth_l1_loss, SmoothL1Loss

from ..builder import ROTATED_LOSSES


@weighted_loss
def probabilistic_l1_loss(pred, target, bbox_conv, beta=1.0):
    """Smooth L1 loss.

    Args:
        bbox_conv:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    bbox_cov = torch.clamp(bbox_conv, -7.0, 7.0)
    loss_box_reg = 0.5 * torch.exp(-bbox_cov) * smooth_l1_loss(
        pred,
        target,
        beta=beta)

    loss_covariance_regularize = 0.5 * bbox_cov
    loss_box_reg += loss_covariance_regularize

    return loss_box_reg


@ROTATED_LOSSES.register_module()
class ProbabilisticL1Loss(nn.Module):
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

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(ProbabilisticL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                bbox_conv=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                compute_bbox_conv=True,
                **kwargs):
        """Smooth L1 loss.

        Args:
            compute_bbox_conv:
            reduction_override:
            avg_factor:
            bbox_conv:
            weight:
            target:
            pred:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if compute_bbox_conv:
            loss_bbox = self.loss_weight * probabilistic_l1_loss(
                pred,
                target,
                bbox_conv,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        else:
            loss_bbox = smooth_l1_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        return loss_bbox
