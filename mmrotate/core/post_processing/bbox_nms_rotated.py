# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import nms_rotated


def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms,
                           max_num=-1,
                           score_factors=None,
                           return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    # Strictly, the maximum coordinates of the rotating box (x,y,w,h,a)
    # should be calculated by polygon coordinates.
    # But the conversion from rbbox to polygon will slow down the speed.
    # So we use max(x,y) + max(w,h) as max coordinate
    # which is larger than polygon max coordinate
    # max(x1, y1, x2, y2,x3, y3, x4, y4)
    max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    if bboxes.size(-1) == 5:
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    else:
        bboxes_for_nms = bboxes + offsets[:, None]
    _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if return_inds:
        return torch.cat([bboxes, scores[:, None]], 1), labels, keep
    else:
        return torch.cat([bboxes, scores[:, None]], 1), labels


def aug_multiclass_nms_rotated(merged_bboxes, merged_labels, score_thr, nms,
                               max_num, classes):
    """NMS for aug multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        classes (int): number of classes.

    Returns:
        tuple (dets, labels): tensors of shape (k, 5), and (k). Dets are boxes
            with scores. Labels are 0-based.
    """
    bboxes, labels = [], []

    for cls in range(classes):
        cls_bboxes = merged_bboxes[merged_labels == cls]
        inds = cls_bboxes[:, -1] > score_thr
        if len(inds) == 0:
            continue
        cur_bboxes = cls_bboxes[inds, :]
        cls_dets, _ = nms_rotated(cur_bboxes[:, :5], cur_bboxes[:, -1],
                                  nms.iou_thr)
        cls_labels = merged_bboxes.new_full((cls_dets.shape[0], ),
                                            cls,
                                            dtype=torch.long)
        if cls_dets.size()[0] == 0:
            continue
        bboxes.append(cls_dets)
        labels.append(cls_labels)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, _inds = bboxes[:, -1].sort(descending=True)
            _inds = _inds[:max_num]
            bboxes = bboxes[_inds]
            labels = labels[_inds]
    else:
        bboxes = merged_bboxes.new_zeros((0, merged_bboxes.size(-1)))
        labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
