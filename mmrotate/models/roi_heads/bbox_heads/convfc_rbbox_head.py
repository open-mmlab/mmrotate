# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.structures.bbox import get_box_tensor
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class RotatedShared2FCBBoxHead(Shared2FCBBoxHead):
    """Rotated Shared2FC RBBox head.

    Args:
        loss_bbox_type (str): Set the input type of ``loss_bbox``.
            Defaults to 'normal'.
    """

    def __init__(self,
                 *args,
                 loss_bbox_type: str = 'normal',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_bbox_type = loss_bbox_type

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox and (self.loss_bbox_type != 'kfiou'):
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                if self.loss_bbox_type == 'normal':
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                elif self.loss_bbox_type == 'kfiou':
                    # When the regression loss (e.g. `KFLoss`)
                    # is applied on both the delta and decoded boxes.
                    bbox_pred_decode = self.bbox_coder.decode(
                        rois[:, 1:], bbox_pred)
                    bbox_pred_decode = get_box_tensor(bbox_pred_decode)
                    bbox_targets_decode = self.bbox_coder.decode(
                        rois[:, 1:], bbox_targets)
                    bbox_targets_decode = get_box_tensor(bbox_targets_decode)

                    if self.reg_class_agnostic:
                        pos_bbox_pred_decode = bbox_pred_decode.view(
                            bbox_pred_decode.size(0),
                            5)[pos_inds.type(torch.bool)]
                    else:
                        pos_bbox_pred_decode = bbox_pred_decode.view(
                            bbox_pred_decode.size(0), -1,
                            5)[pos_inds.type(torch.bool),
                               labels[pos_inds.type(torch.bool)]]

                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        pred_decode=pos_bbox_pred_decode,
                        targets_decode=bbox_targets_decode[pos_inds.type(
                            torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                else:
                    raise NotImplementedError
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
