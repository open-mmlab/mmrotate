# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models import weighted_loss
from torch import nn

from mmrotate.registry import MODELS


def gbb_form(boxes):
    return torch.cat(
        (boxes[:, :2], torch.pow(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), 1)


def rotated_form(a_, b_, angles):
    a = a_ * torch.pow(torch.cos(angles), 2.) + b_ * torch.pow(
        torch.sin(angles), 2.)
    b = a_ * torch.pow(torch.sin(angles), 2.) + b_ * torch.pow(
        torch.cos(angles), 2.)
    c = a_ * torch.cos(angles) * torch.sin(angles) - b_ * torch.sin(
        angles) * torch.cos(angles)
    return a, b, c


@weighted_loss
def probiou_loss(pred, target, eps=1e-3, mode='l1'):
    """pred    -> a matrix [N,5](x,y,w,h,angle) containing ours predicted box
    target  -> a matrix [N,5](x,y,w,h,angle) containing ours target    box eps.

    -> threshold to avoid infinite values mode    -> ('l1' in [0,1] or 'l2' in
    [0,inf]) metrics according our paper.
    """

    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    (x1, y1, a1_, b1_, c1_) = (gbboxes1[:, 0], gbboxes1[:, 1], gbboxes1[:, 2],
                               gbboxes1[:, 3], gbboxes1[:, 4])
    (x2, y2, a2_, b2_, c2_) = (gbboxes2[:, 0], gbboxes2[:, 1], gbboxes2[:, 2],
                               gbboxes2[:, 3], gbboxes2[:, 4])

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)

    t1 = (((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) *
           (torch.pow(x1 - x2, 2))) / ((a1 + a2) * (b1 + b2) -
                                       (torch.pow(c1 + c2, 2)) + eps)) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) /
          ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = torch.log(((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2))) /
                   (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)) *
                                   (a2 * b2 - torch.pow(c2, 2))) + eps) +
                   eps) * 0.5

    B_d = t1 + t2 + t3

    B_d = torch.clamp(B_d, eps, 100.0)
    l1 = torch.sqrt(1.0 - torch.exp(-B_d) + eps)
    l_i = torch.pow(l1, 2.0)
    l2 = -torch.log(1.0 - l_i + eps)

    if mode == 'l1':
        probiou = l1
    if mode == 'l2':
        probiou = l2

    return probiou


@MODELS.register_module()
class ProbIoULoss(nn.Module):
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

    def __init__(self, mode='l1', eps=1e-6, reduction='mean', loss_weight=1.0):
        super(ProbIoULoss, self).__init__()

        self.mode = mode
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
        loss = self.loss_weight * probiou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
