# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes, register_bbox_mode
from torch import BoolTensor, Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]


@register_bbox_mode('qbox')
class QuadriBoxes(BaseBoxes):
    """The quadrilateral box class.

    The ``_bbox_dim`` of ``QuadriBoxes`` is 8, which means the length of the
    last dimension of the input should be 8. Each row of data means (x1, y1,
    x2, y2, x3, y3, x4, y4) which are the coordinates of 4 vertices of the box.

    ``QuadriBoxes`` usually works as the raw data loaded from dataset like
    DOTA, DIOR, etc.

    Args:
        bboxes (Tensor or np.ndarray or Sequence): The box data with
            shape (..., 8).
        dtype (torch.dtype, Optional): data type of bboxes. Defaults to None.
        device (str or torch.device, Optional): device of bboxes.
            Default to None.
        clone (bool): Whether clone ``bboxes`` or not. Defaults to True.
    """

    _bbox_dim = 8

    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes."""
        bboxes = self.tensor
        bboxes = bboxes.reshape(*bboxes.shape[:-1], 4, 2)
        return bboxes.mean(dim=-2)

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes."""
        bboxes = self.tensor
        pts = bboxes.reshape(*bboxes.shape[:-1], 4, 2)
        roll_pts = torch.roll(pts, 1, dims=-2)
        xyxy = torch.sum(
            pts[..., 0] * roll_pts[..., 1] - roll_pts[..., 0] * pts[..., 1],
            dim=-1)
        areas = 0.5 * torch.abs(xyxy)
        return areas

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes.

        Quadrilateral boxes don't have the width concept. Use ``sqrt(areas)``
        to replace the width.
        """
        return torch.sqrt(self.areas)

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes.

        Quadrilateral boxes don't have the height concept. Use ``sqrt(areas)``
        to replace the heights.
        """
        return torch.sqrt(self.areas)

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Inplace flip bboxes horizontally or vertically.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        flipped = self.tensor
        if direction == 'horizontal':
            flipped[..., 0::2] = img_shape[1] - flipped[..., 0::2]
        elif direction == 'vertical':
            flipped[..., 1::2] = img_shape[0] - flipped[..., 1::2]
        else:
            flipped[..., 0::2] = img_shape[1] - flipped[..., 0::2]
            flipped[..., 1::2] = img_shape[0] - flipped[..., 1::2]

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Inplace translate bboxes.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        bboxes = self.tensor
        assert len(distances) == 2
        self.tensor = bboxes + bboxes.new_tensor(distances).repeat(4)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Inplace clip boxes according to the image shape.

        In ``QuadriBoxes``, ``clip`` function only clones the original data,
        because it's very tricky to handle quadrilateral boxes corssing the
        image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            T: Cliped boxes with the same shape as the original boxes.
        """
        pass

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Inplace rotate all boxes.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees.
        """
        bboxes = self.tensor
        rotation_matrix = bboxes.new_tensor(
            cv2.getRotationMatrix2D(center, angle, 1))

        corners = bboxes.reshape(*bboxes.shape[:-1], 4, 2)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(rotation_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        self.tensor = corners.reshape(*corners.shape[:-2], 8)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Inplace geometric transformation for bbox.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        bboxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = bboxes.new_tensor(homography_matrix)
        corners = bboxes.reshape(*bboxes.shape[:-1], 4, 2)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = corners.reshape(*corners.shape[:-2], 8)

    def rescale_(self,
                 scale_factor: Tuple[float, float],
                 mapping_back=False) -> None:
        """Inplace rescale boxes w.r.t. rescale_factor.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink bboxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of bboxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
            mapping_back (bool): Mapping back the rescaled bboxes.
                Defaults to False.
        """
        bboxes = self.tensor
        assert len(scale_factor) == 2
        scale_factor = bboxes.new_tensor(scale_factor).repeat(4)
        self.tensor = bboxes / scale_factor if mapping_back else \
            bboxes * scale_factor

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Inplace resize the box width and height w.r.t scale_factor.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink bboxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of bboxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        bboxes = self.tensor
        assert len(scale_factor) == 2
        assert scale_factor[0] == scale_factor[1], \
            'To protect the shape of QuadriBoxes not changes'
        scale_factor = bboxes.new_tensor(scale_factor)

        bboxes = bboxes.reshape(*bboxes.shape[:-1], 4, 2)
        centers = bboxes.mean(dim=-2)[..., None, :]
        bboxes = (bboxes - centers) * scale_factor + centers
        self.tensor = bboxes.reshape(*bboxes.shape[:-2], 8)

    def is_bboxes_inside(self, img_shape: Tuple[int, int]) -> BoolTensor:
        """Find bboxes inside the image.

        In ``QuadriBoxes``, as long as the center of box is inside the
        image, this box will be regarded as True.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            BoolTensor: Index of the remaining bboxes. Assuming the original
            quadrilateral boxes have shape (m, n, 8), the output has shape
            (m, n).
        """
        img_h, img_w = img_shape
        bboxes = self.tensor
        bboxes = bboxes.reshape(*bboxes.shape[:-1], 4, 2)
        centers = bboxes.mean(dim=-2)
        return (centers[..., 0] < img_w) & (centers[..., 0] > 0) \
            & (centers[..., 1] < img_h) & (centers[..., 1] > 0)

    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False) -> BoolTensor:
        """Find inside box points. Require bboxes dimension must be 2.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with bboxes
                or not. If True, the length of bboxes and ``points`` should be
                the same. Defaults to False.

        Returns:
            BoolTensor: Index of inside box points. Assuming the boxes has
            shape of (n, 8), if ``is_aligned`` is False. The index has
            shape of (m, n). If ``is_aligned`` is True, m should be equal to n
            and the index has shape of (m, ).
        """
        bboxes = self.tensor
        assert bboxes.dim() == 2, 'bboxes dimension must be 2.'

        corners = bboxes.reshape(-1, 4, 2)
        corners_next = torch.roll(corners, -1, dims=1)
        x1, y1 = corners.unbind(dim=2)
        x2, y2 = corners_next.unbind(dim=2)
        pt_x, pt_y = points.split([1, 1], dim=1)

        if not is_aligned:
            pt_x = pt_x[:, None, :]
            pt_y = pt_y[:, None, :]
            x1 = x1[None, :, :]
            y1 = y1[None, :, :]
            x2 = x2[None, :, :]
            y2 = y2[None, :, :]
        else:
            assert bboxes.size(0) == points.size(0)

        values = (x1 - pt_x) * (y2 - pt_y) - (y1 - pt_y) * (x2 - pt_x)
        return (values > 0).all(dim=-1) | (values < 0).all(dim=-1)
