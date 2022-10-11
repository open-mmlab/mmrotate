# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from mmrotate.registry import MODELS


def smooth_focal_loss(pred,
                      target,
                      weight=None,
                      gamma=2.0,
                      alpha=0.25,
                      reduction='mean',
                      avg_factor=None):
    """Smooth Focal Loss proposed in Circular Smooth Label (CSL).

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The reduction method used to
            override the original reduction method of the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class SmoothFocalLoss(nn.Module):
    """Smooth Focal Loss. Implementation of `Circular Smooth Label (CSL).`__

    __ https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SmoothFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * smooth_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls
