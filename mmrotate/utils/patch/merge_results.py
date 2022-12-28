# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import numpy as np
from mmcv.ops import batched_nms
from mmdet.structures import DetDataSample, SampleList
from mmengine.structures import InstanceData
from torch import Tensor


def translate_bboxes(bboxes: Tensor, offset: Sequence[int]):
    """Translate bboxes w.r.t offset.

    The bboxes can be three types:

    - HorizontalBoxes: The boxes should be a tensor with shape of (n, 4),
      which means (x, y, x, y).
    - RotatedBoxes: The boxes should be a tensor with shape of (n, 5),
      which means (x, y, w, h, t).
    - QuariBoxes: The boxes should be a tensor with shape of (n, 8),
      which means (x1, y1, x2, y2, x3, y3, x4, y4).

    Args:
        bboxes (Tensor): The bboxes need to be translated. Its shape can
            be (n, 4), (n, 5), or (n, 8).
        offset (Sequence[int]): The translation offsets with shape of (2, ).

    Returns:
        Tensor: Translated bboxes.
    """
    if bboxes.shape[1] == 4:
        offset = bboxes.new_tensor(offset).tile(2)
        bboxes = bboxes + offset
    elif bboxes.shape[1] == 5:
        offset = bboxes.new_tensor(offset)
        bboxes[:, :2] = bboxes[:, :2] + offset
    elif bboxes.shape[1] == 8:
        offset = bboxes.new_tensor(offset).tile(4)
        bboxes = bboxes + offset
    else:
        raise TypeError('Require the shape of `bboxes` to be (n, 5), (n, 6)'
                        'or (n, 8), but get `bboxes` with shape being '
                        f'{bboxes.shape}.')
    return bboxes


def map_masks(masks: np.ndarray, offset: Sequence[int],
              new_shape: Sequence[int]) -> np.ndarray:
    """Map masks to the huge image.

    Args:
        masks (:obj:`np.ndarray`): masks need to be mapped.
        offset (Sequence[int]): The offset to translate with shape of (2, ).
        new_shape (Sequence[int]): A tuple of the huge image's width
            and height.

    Returns:
        :obj:`np.ndarray`: Mapped masks.
    """
    # empty masks
    if not masks:
        return masks

    new_width, new_height = new_shape
    x_start, y_start = offset
    mapped = []
    for mask in masks:
        ori_height, ori_width = mask.shape[:2]

        x_end = x_start + ori_width
        if x_end > new_width:
            ori_width -= x_end - new_width
            x_end = new_width

        y_end = y_start + ori_height
        if y_end > new_height:
            ori_height -= y_end - new_height
            y_end = new_height

        extended_mask = np.zeros((new_height, new_width), dtype=bool)
        extended_mask[y_start:y_end,
                      x_start:x_end] = mask[:ori_height, :ori_width]
        mapped.append(extended_mask)
    return np.stack(mapped, axis=0)


def merge_results_by_nms(results: SampleList, offsets: np.ndarray,
                         img_shape: Tuple[int, int],
                         nms_cfg: dict) -> DetDataSample:
    """Merge patch results by nms.

    Args:
        results (List[:obj:`DetDataSample`]): A list of patches results.
        offsets (:obj:`np.ndarray`): Positions of the left top points
            of patches.
        img_shape (Tuple[int, int]): A tuple of the huge image's width
            and height.
        nms_cfg (dict): it should specify nms type and other parameters
            like `iou_threshold`.

    Retunrns:
        :obj:`DetDataSample`: merged results.
    """
    assert len(results) == offsets.shape[0], 'The `results` should has the ' \
                                             'same length with `offsets`.'
    pred_instances = []
    for result, offset in zip(results, offsets):
        pred_inst = result.pred_instances
        pred_inst.bboxes = translate_bboxes(pred_inst.bboxes, offset)
        if 'masks' in result:
            pred_inst.masks = map_masks(pred_inst.masks, offset, img_shape)
        pred_instances.append(pred_inst)

    instances = InstanceData.cat(pred_instances)
    _, keeps = batched_nms(
        boxes=instances.bboxes,
        scores=instances.scores,
        idxs=instances.labels,
        nms_cfg=nms_cfg)
    merged_instances = instances[keeps]

    merged_result = DetDataSample()
    # update items like gt_instances, ignore_instances
    merged_result.update(results[0])
    merged_result.pred_instances = merged_instances
    return merged_result
