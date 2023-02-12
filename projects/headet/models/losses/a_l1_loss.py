# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mmrotate.registry import MODELS
from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def al1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()



    # pred_part1, pred_part2 = torch.split(pred, [4, 1], dim=1)
    # target_part1, target_part2 = torch.split(target, [4, 1], dim=1)


    # loss_part1 = torch.abs(pred_part1-target_part1)
    pred = (pred + np.pi) % (2*np.pi) - np.pi 

    assert pred.max() < np.pi
    assert pred.min() > -np.pi
    assert target.max() < np.pi
    assert target.min() > -np.pi



    # pred = torch.where(pred > torch.tensor(np.pi, dtype=torch.float).cuda(), torch.tensor(np.pi, dtype=torch.float).cuda(), pred)
    # target = torch.where(target < -torch.tensor(np.pi, dtype=torch.float).cuda(), torch.tensor(np.pi, dtype=torch.float).cuda(), target)


    # pred_part2 = torch.where(pred_part2 > np.pi, np.pi, pred_part2.float())
    # pred_part2 = torch.where(pred_part2 < -np.pi, -np.pi, pred_part2.float())
    loss = torch.abs(pred-target)

    # loss_part2_1 = torch.abs((pred_part2-target_part2))
    loss = torch.where(loss > np.pi, 2 *
                             np.pi-loss, loss)

    # print(loss_part2.sum())
    # print(loss_part1.mean(), loss_part2.mean())

    # loss = torch.cat((loss_part1, loss_part2), dim=1)

    return loss


@MODELS.register_module()
class AL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * al1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
