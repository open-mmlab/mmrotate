# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.structures.bbox import (HorizontalBoxes, bbox_overlaps,
                                   get_box_tensor)
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import (QuadriBoxes, RotatedBoxes,
                                      fake_rbbox_overlaps, rbbox_overlaps)


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
                 bboxes2: HorizontalBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Convert gt from rbb to hbb, and calculate IoU between hbboxes.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`HorizontalBoxes` or Tensor): bboxes have shape
                (n, 4) in <x1, y1, x2, y2> format, shape (n, 5) in
                <x1, y1, x2, y2, score> format, or be empty.
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


@TASK_UTILS.register_module()
class QBbox2HBboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale: float = 1., dtype: str = None) -> None:
        self.scale = scale
        self.dtype = dtype

    def __call__(self,
                 bboxes1: QuadriBoxes,
                 bboxes2: HorizontalBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Convert gt from qbb to hbb, and calculate IoU between hbboxes.

        Args:
            bboxes1 (:obj:`QuadriBoxes` or Tensor): bboxes have shape (m, 8)
                in <x1, y1, ..., x4, y4> format, shape (m, 9) in
                <x1, y1, ..., x4, y4, score> format.
            bboxes2 (:obj:`HorizontalBoxes` or Tensor): bboxes have shape
                (n, 4) in <x1, y1, x2, y2> format, shape (n, 5) in
                <x1, y1, x2, y2, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground).
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 8, 9]
        assert bboxes2.size(-1) in [0, 4, 5]

        if bboxes1.size(-1) == 9:
            bboxes1 = bboxes1[..., :8]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]

        if not isinstance(bboxes1, QuadriBoxes):
            bboxes1 = QuadriBoxes(bboxes1)
        # convert qbb to minimum circumscribed hbb in <x1, y1, x2, y2> format.
        if bboxes1.size(0) == 0:
            bboxes1 = bboxes1.new_zeros(0, 4)
        else:
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
