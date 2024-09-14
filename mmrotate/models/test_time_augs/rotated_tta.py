# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch

from torch import Tensor

from mmdet.models.test_time_augs import DetTTAModel
from mmrotate.registry import MODELS


def bbox_flip(bboxes: Tensor,
              img_shape: Tuple[int],
              direction: str = 'horizontal') -> Tensor:
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 5*k)
        img_shape (Tuple[int]): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 5 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0] = img_shape[1] - flipped[..., 0]
        flipped[..., 4] = -flipped[..., 4]
    elif direction == 'vertical':
        flipped[..., 1] = img_shape[0] - flipped[..., 1]
        flipped[..., 4] = -flipped[..., 4]
    else:
        flipped[..., 0] = img_shape[1] - flipped[..., 0]
        flipped[..., 1] = img_shape[0] - flipped[..., 1]
    return flipped

@MODELS.register_module()
class RotatedTTAModel(DetTTAModel):

    def merge_aug_bboxes(self, aug_bboxes: List[Tensor],
                         aug_scores: List[Tensor],
                         img_metas: List[str]) -> Tuple[Tensor, Tensor]:
        """Merge augmented detection bboxes and scores.
        Args:
            aug_bboxes (list[Tensor]): shape (n, 5*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,5), where
            4 represent (x, y, w, h, t)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            ori_shape = img_info['ori_shape']
            flip = img_info['flip']
            flip_direction = img_info['flip_direction']
            if flip:
                bboxes = bbox_flip(
                    bboxes=bboxes,
                    img_shape=ori_shape,
                    direction=flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores