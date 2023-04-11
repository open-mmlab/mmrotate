# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.utils import ConfigType
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class H2RBoxV2ConsistencyLoss(torch.nn.Module):

    def __init__(self,
                 loss_rot: ConfigType = dict(
                     type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
                 loss_flp: ConfigType = dict(
                     type='mmdet.SmoothL1Loss', loss_weight=0.05, beta=0.1),
                 use_snap_loss: bool = True,
                 reduction: str = 'mean') -> None:
        super(H2RBoxV2ConsistencyLoss, self).__init__()
        self.loss_rot = MODELS.build(loss_rot)
        self.loss_flp = MODELS.build(loss_flp)
        self.use_snap_loss = use_snap_loss
        self.reduction = reduction

    def forward(self,
                pred_ori: Tensor,
                pred_rot: Tensor,
                pred_flp: Tensor,
                target_ori: Tensor,
                target_rot: Tensor,
                agnostic_mask: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted boxes.
            target (Tensor): Corresponding gt boxes.
            weight (Tensor): The weight of loss for each prediction.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            Calculated loss (Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        d_ang_rot = (pred_ori - pred_rot) - (target_ori - target_rot)
        d_ang_flp = pred_ori + pred_flp

        if self.use_snap_loss:
            d_ang_rot = (d_ang_rot + torch.pi / 2) % torch.pi - torch.pi / 2
            d_ang_flp = (d_ang_flp + torch.pi / 2) % torch.pi - torch.pi / 2

        if agnostic_mask is not None:
            d_ang_rot[agnostic_mask] = 0
            d_ang_flp[agnostic_mask] = 0

        loss_rot = self.loss_rot(
            d_ang_rot,
            torch.zeros_like(d_ang_rot),
            reduction_override=reduction,
            avg_factor=avg_factor)
        loss_flp = self.loss_flp(
            d_ang_flp,
            torch.zeros_like(d_ang_flp),
            reduction_override=reduction,
            avg_factor=avg_factor)

        return loss_rot + loss_flp
