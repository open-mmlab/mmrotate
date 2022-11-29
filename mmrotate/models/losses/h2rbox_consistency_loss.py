# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.utils import ConfigType
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class H2RBoxConsistencyLoss(torch.nn.Module):

    def __init__(self,
                 center_loss_cfg: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=0.0),
                 shape_loss_cfg: ConfigType = dict(
                     type='mmdet.IoULoss', loss_weight=1.0),
                 angle_loss_cfg: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=1.0),
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super(H2RBoxConsistencyLoss, self).__init__()
        self.center_loss = MODELS.build(center_loss_cfg)
        self.shape_loss = MODELS.build(shape_loss_cfg)
        self.angle_loss = MODELS.build(angle_loss_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Tensor,
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
        xy_pred = pred[..., :2]
        xy_target = target[..., :2]
        hbb_pred1 = torch.cat([-pred[..., 2:4], pred[..., 2:4]], dim=-1)
        hbb_pred2 = hbb_pred1[..., [1, 0, 3, 2]]
        hbb_target = torch.cat([-target[..., 2:4], target[..., 2:4]], dim=-1)
        d_a_pred = pred[..., 4] - target[..., 4]

        center_loss = self.center_loss(
            xy_pred,
            xy_target,
            weight=weight[:, None],
            reduction_override=reduction,
            avg_factor=avg_factor)
        shape_loss1 = self.shape_loss(
            hbb_pred1,
            hbb_target,
            weight=weight,
            reduction_override=reduction,
            avg_factor=avg_factor) + self.angle_loss(
                d_a_pred.sin(),
                torch.zeros_like(d_a_pred),
                weight=weight,
                reduction_override=reduction,
                avg_factor=avg_factor)
        shape_loss2 = self.shape_loss(
            hbb_pred2,
            hbb_target,
            weight=weight,
            reduction_override=reduction,
            avg_factor=avg_factor) + self.angle_loss(
                d_a_pred.cos(),
                torch.zeros_like(d_a_pred),
                weight=weight,
                reduction_override=reduction,
                avg_factor=avg_factor)
        loss_bbox = center_loss + torch.min(shape_loss1, shape_loss2)
        return self.loss_weight * loss_bbox
