# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import box_iou_rotated
from mmdet.structures.bbox import BaseBoxes, bbox_overlaps, get_box_tensor
from torch import Tensor

from mmrotate.core.bbox.structures import RotatedBoxes
from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class RBboxOverlaps2D(object):
    """2D Overlaps Calculator for Rotated Bboxes."""

    def __call__(self,
                 bboxes1: RotatedBoxes,
                 bboxes2: RotatedBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Calculate IoU between 2D rotated bboxes.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (n, 5)
                in <cx, cy, w, h, t> format, shape (n, 6) in
                <cx, cy, w, h, t, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground). Defaults to 'iou'.
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]

        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]

        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)

        return rbbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


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

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)


@TASK_UTILS.register_module()
class FakeRBboxOverlaps2D(object):
    """2D Overlaps Calculator for Minimum Circumscribed Horizental Bboxes of
    Rotated Bboxes."""

    def __call__(self,
                 bboxes1: RotatedBoxes,
                 bboxes2: RotatedBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Calculate IoU between 2D minimum circumscribed hbbs of rbbs.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (n, 5)
                in <cx, cy, w, h, t> format, shape (n, 6) in
                <cx, cy, w, h, t, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground).
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]

        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]

        if not isinstance(bboxes1, RotatedBoxes):
            bboxes1 = RotatedBoxes(bboxes1)
        if not isinstance(bboxes2, RotatedBoxes):
            bboxes2 = RotatedBoxes(bboxes2)

        return fake_rbbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


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
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    # convert rbb to minimum circumscribed hbb in <cx, cy, w, h, t> format.
    fake_rbboxes1 = bboxes1.convert_to('hbox').convert_to('rbox')
    fake_rbboxes2 = bboxes2.convert_to('hbox').convert_to('rbox')

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = fake_rbboxes1.detach().clone().tensor
    clamped_bboxes2 = fake_rbboxes2.detach().clone().tensor
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)


def cast_tensor_type(x: Tensor,
                     scale: float = 1.,
                     dtype: str = None) -> Tensor:
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class RBbox2HBboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale: float = 1., dtype: str = None) -> None:
        self.scale = scale
        self.dtype = dtype

    def __call__(self,
                 bboxes1: RotatedBoxes,
                 bboxes2: BaseBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Convert gt from rbb to hbb, and calculate IoU between hbboxes.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (n, 5)
                in <cx, cy, w, h, t> format, shape (n, 6) in
                <cx, cy, w, h, t, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground).
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 4, 5]

        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]

        if not isinstance(bboxes1, RotatedBoxes):
            bboxes1 = RotatedBoxes(bboxes1)
        # convert rbb to minimum circumscribed hbb in <x1, y1, x2, y2> format.
        bboxes1 = bboxes1.convert_to('hbox').tensor
        bboxes2 = get_box_tensor(bboxes2)

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str
