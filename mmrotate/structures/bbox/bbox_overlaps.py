# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import box_iou_rotated
from torch import Tensor

from .rotated_boxes import RotatedBoxes


def rbbox_overlaps(bboxes1: Tensor,
                   bboxes2: Tensor,
                   mode: str = 'iou',
                   is_aligned: bool = False) -> Tensor:
    """Calculate overlap between two set of rotated bboxes.

    Args:
        bboxes1 (Tensor): shape (B, m, 5) in <cx, cy, w, h, t> format
            or empty.
        bboxes2 (Tensor): shape (B, n, 5) in <cx, cy, w, h, t> format
            or empty.
        mode (str): 'iou' (intersection over union), 'iof' (intersection over
            foreground). Defaults to 'iou'.
        is_aligned (bool): If True, then m and n must be equal.
            Defaults to False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = bboxes1.detach().clone()
    clamped_bboxes2 = bboxes2.detach().clone()
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    # resolve `rbbox_overlaps` abnormal when coordinate value is too large.
    # TODO: fix in mmcv
    clamped_bboxes1[:, :2].clamp_(min=-1e7, max=1e7)
    clamped_bboxes2[:, :2].clamp_(min=-1e7, max=1e7)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)


def fake_rbbox_overlaps(bboxes1: RotatedBoxes,
                        bboxes2: RotatedBoxes,
                        mode: str = 'iou',
                        is_aligned: bool = False) -> Tensor:
    """Calculate overlap between two set of minimum circumscribed hbbs of rbbs.

    Args:
        bboxes1 (:obj:`RotatedBoxes`): shape (B, m, 5) in <cx, cy, w, h, t>
            format or empty.
        bboxes2 (:obj:`RotatedBoxes`): shape (B, n, 5) in <cx, cy, w, h, t>
            format or empty.
        mode (str): 'iou' (intersection over union), 'iof' (intersection over
            foreground).
            Defaults to 'iou'.
        is_aligned (bool): If True, then m and n must be equal.
            Defaults to False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.tensor.new(
            rows, 1) if is_aligned else bboxes1.tensor.new(rows, cols)

    # convert rbb to minimum circumscribed hbb in <cx, cy, w, h, t> format.
    fake_rbboxes1 = bboxes1.convert_to('hbox').convert_to('rbox')
    fake_rbboxes2 = bboxes2.convert_to('hbox').convert_to('rbox')

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = fake_rbboxes1.detach().clone().tensor
    clamped_bboxes2 = fake_rbboxes2.detach().clone().tensor
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    # resolve `rbbox_overlaps` abnormal when coordinate value is too large.
    # TODO: fix in mmcv
    clamped_bboxes1[:, :2].clamp_(min=-1e7, max=1e7)
    clamped_bboxes2[:, :2].clamp_(min=-1e7, max=1e7)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)
