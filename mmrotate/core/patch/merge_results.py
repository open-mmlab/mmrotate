# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.ops import nms, nms_rotated


def translate_bboxes(bboxes, offset):
    """Translate bboxes according to its shape.

    If the bbox shape is (n, 5), the bboxes are regarded as horizontal bboxes
    and in (x, y, x, y, score) format. If the bbox shape is (n, 6), the bboxes
    are regarded as rotated bboxes and in (x, y, w, h, theta, score) format.

    Args:
        bboxes (np.ndarray): The bboxes need to be translated. Its shape can
            only be (n, 5) and (n, 6).
        offset (np.ndarray): The offset to translate with shape being (2, ).

    Returns:
        np.ndarray: Translated bboxes.
    """
    if bboxes.shape[1] == 5:
        bboxes[:, :4] = bboxes[:, :4] + np.tile(offset, 2)
    elif bboxes.shape[1] == 6:
        bboxes[:, :2] = bboxes[:, :2] + offset
    else:
        raise TypeError('Require the shape of `bboxes` to be (n, 5) or (n, 6),'
                        f' but get `bboxes` with shape being {bboxes.shape}.')
    return bboxes


def map_masks(masks, offset, new_shape):
    """Map masks to the huge image.

    Args:
        masks (list[np.ndarray]): masks need to be mapped.
        offset (np.ndarray): The offset to translate with shape being (2, ).
        new_shape (tuple): A tuple of the huge image's width and height.

    Returns:
        list[np.ndarray]: Mapped masks.
    """
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

        extended_mask = np.zeros((new_height, new_width), dtype=np.bool)
        extended_mask[y_start:y_end,
                      x_start:x_end] = mask[:ori_height, :ori_width]
        mapped.append(extended_mask)
    return mapped


def merge_results(results, offsets, img_shape, iou_thr=0.1, device='cpu'):
    """Merge patch results via nms.

    Args:
        results (list[np.ndarray] | list[tuple]): A list of patches results.
        offsets (np.ndarray): Positions of the left top points of patches.
        img_shape (tuple): A tuple of the huge image's width and height.
        iou_thr (float): The IoU threshold of NMS.
        device (str): The device to call nms.

    Retunrns:
        list[np.ndarray]: Detection results after merging.
    """
    assert len(results) == offsets.shape[0], 'The `results` should has the ' \
                                             'same length with `offsets`.'
    with_mask = isinstance(results[0], tuple)
    num_patches = len(results)
    num_classes = len(results[0][0]) if with_mask else len(results[0])

    merged_bboxes = []
    merged_masks = []
    for cls in range(num_classes):
        if with_mask:
            dets_per_cls = [results[i][0][cls] for i in range(num_patches)]
            masks_per_cls = [results[i][1][cls] for i in range(num_patches)]
        else:
            dets_per_cls = [results[i][cls] for i in range(num_patches)]
            masks_per_cls = None

        dets_per_cls = [
            translate_bboxes(dets_per_cls[i], offsets[i])
            for i in range(num_patches)
        ]
        dets_per_cls = np.concatenate(dets_per_cls, axis=0)
        if with_mask:
            masks_placeholder = []
            for i, masks in enumerate(masks_per_cls):
                translated = map_masks(masks, offsets[i], img_shape)
                masks_placeholder.extend(translated)
            masks_per_cls = masks_placeholder

        if dets_per_cls.size == 0:
            merged_bboxes.append(dets_per_cls)
            if with_mask:
                merged_masks.append(masks_per_cls)
        else:
            dets_per_cls = torch.from_numpy(dets_per_cls).to(device)
            nms_func = nms if dets_per_cls.size(1) == 5 else nms_rotated
            nms_dets, keeps = nms_func(dets_per_cls[:, :-1],
                                       dets_per_cls[:, -1], iou_thr)
            merged_bboxes.append(nms_dets.cpu().numpy())
            if with_mask:
                keeps = keeps.cpu().numpy()
                merged_masks.append([masks_per_cls[i] for i in keeps])

    if with_mask:
        return merged_bboxes, merged_masks
    else:
        return merged_bboxes
