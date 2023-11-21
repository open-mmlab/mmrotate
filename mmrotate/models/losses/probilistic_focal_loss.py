# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models.losses import sigmoid_focal_loss

from .smooth_focal_loss import smooth_focal_loss
from ..builder import ROTATED_LOSSES


def probabilistic_focal_loss(pred,
                             target,
                             logits_var,
                             weight=None,
                             gamma=2.0,
                             alpha=0.25,
                             reduction='mean',
                             avg_factor=None,
                             num_samples=10):
    """Smooth Focal Loss proposed in Circular Smooth Label (CSL).

    Args:
        num_samples:
        logits_var:
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

    logits_var = torch.sqrt(torch.exp(logits_var))
    univariate_normal_dists = torch.distributions.normal.Normal(pred, scale=logits_var)

    stochastic_logits = univariate_normal_dists.rsample(
        (num_samples,))
    stochastic_logits = stochastic_logits.view(
        (stochastic_logits.shape[1] * num_samples, stochastic_logits.shape[2], -1))
    stochastic_logits = stochastic_logits.squeeze(2)

    target = torch.unsqueeze(target, 0)
    target = torch.repeat_interleave(target, num_samples, dim=0).view(
        (target.shape[1] * num_samples, target.shape[2], -1))
    target = target.squeeze(2)

    loss = sigmoid_focal_loss(
        stochastic_logits,
        target,
        weight=weight,
        gamma=gamma,
        alpha=alpha,
        reduction=reduction,
        avg_factor=avg_factor)
    return loss


@ROTATED_LOSSES.register_module()
class ProbabilisticFocalLoss(nn.Module):
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
        super(ProbabilisticFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                logits_var=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                compute_cls_var=True):
        """Forward function.

        Args:
            logits_var:
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

        if compute_cls_var:
            loss_cls = self.loss_weight * probabilistic_focal_loss(
                pred,
                target,
                logits_var,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            loss_cls = sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor
            )

        return loss_cls
