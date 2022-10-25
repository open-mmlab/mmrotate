# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes, register_box
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torch import BoolTensor, Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]


@register_box('qbox')
class QuadriBoxes(BaseBoxes):
    """The quadrilateral box class.

    The ``box_dim`` of ``QuadriBoxes`` is 8, which means the length of the
    last dimension of the input should be 8. Each row of data means (x1, y1,
    x2, y2, x3, y3, x4, y4) which are the coordinates of 4 vertices of the box.
    The box must be convex. The order of 4 vertices can be both CW and CCW.

    ``QuadriBoxes`` usually works as the raw data loaded from dataset like
    DOTA, DIOR, etc.

    Args:
        boxes (Tensor or np.ndarray or Sequence): The box data with
            shape (..., 8).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    box_dim = 8

    @property
    def vertices(self) -> Tensor:
        """Return a tensor representing the vertices of boxes.

        If boxes have shape of (m, 8), vertices have shape of (m, 4, 2)
        """
        boxes = self.tensor
        return boxes.reshape(*boxes.shape[:-1], 4, 2)

    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes.

        If boxes have shape of (m, 8), centers have shape of (m, 2).
        """
        boxes = self.tensor
        boxes = boxes.reshape(*boxes.shape[:-1], 4, 2)
        return boxes.mean(dim=-2)

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes.

        If boxes have shape of (m, 8), areas have shape of (m, ).
        """
        boxes = self.tensor
        pts = boxes.reshape(*boxes.shape[:-1], 4, 2)
        roll_pts = torch.roll(pts, 1, dims=-2)
        xyxy = torch.sum(
            pts[..., 0] * roll_pts[..., 1] - roll_pts[..., 0] * pts[..., 1],
            dim=-1)
        areas = 0.5 * torch.abs(xyxy)
        return areas

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes.

        If boxes have shape of (m, 8), widths have shape of (m, ).

        notes:
            Quadrilateral boxes don't have the width concept. Use
            ``sqrt(areas)`` to replace the width.
        """
        warnings.warn("Quadrilateral boxes don't have the width concept. "
                      'We use ``sqrt(areas)`` to replace the width.')
        return torch.sqrt(self.areas)

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes.

        If boxes have shape of (m, 8), heights have shape of (m, ).

        notes:
            Quadrilateral boxes don't have the height concept. Use
            ``sqrt(areas)`` to replace the heights.
        """
        warnings.warn("Quadrilateral boxes don't have the height concept. "
                      'We use ``sqrt(areas)`` to replace the width.')
        return torch.sqrt(self.areas)

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.

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
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        boxes = self.tensor
        assert len(distances) == 2
        self.tensor = boxes + boxes.new_tensor(distances).repeat(4)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        In ``QuadriBoxes``, ``clip`` function does nothing about the original
        data, because it's very tricky to handle rotate boxes corssing the
        image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            T: Cliped boxes with the same shape as the original boxes.
        """
        warnings.warn('The `clip` function does nothing in `QuadriBoxes`.')

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        boxes = self.tensor
        rotation_matrix = boxes.new_tensor(
            cv2.getRotationMatrix2D(center, -angle, 1))

        corners = boxes.reshape(*boxes.shape[:-1], 4, 2)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(rotation_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        self.tensor = corners.reshape(*corners.shape[:-2], 8)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        boxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)
        corners = boxes.reshape(*boxes.shape[:-1], 4, 2)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = corners.reshape(*corners.shape[:-2], 8)

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        boxes = self.tensor
        assert len(scale_factor) == 2
        scale_factor = boxes.new_tensor(scale_factor).repeat(4)
        self.tensor = boxes * scale_factor

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Resize the box width and height w.r.t scale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        boxes = self.tensor
        assert len(scale_factor) == 2
        assert scale_factor[0] == scale_factor[1], \
            'To protect the shape of QuadriBoxes not changes'
        scale_factor = boxes.new_tensor(scale_factor)

        boxes = boxes.reshape(*boxes.shape[:-1], 4, 2)
        centers = boxes.mean(dim=-2)[..., None, :]
        boxes = (boxes - centers) * scale_factor + centers
        self.tensor = boxes.reshape(*boxes.shape[:-2], 8)

    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.

        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 8),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        boxes = self.tensor
        boxes = boxes.reshape(*boxes.shape[:-1], 4, 2)
        centers = boxes.mean(dim=-2)
        return (centers[..., 0] <= img_w + allowed_border) & \
               (centers[..., 1] <= img_h + allowed_border) & \
               (centers[..., 0] >= -allowed_border) & \
               (centers[..., 1] >= -allowed_border)

    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False,
                           eps: float = 0.01) -> BoolTensor:
        """Find inside box points. Boxes dimension must be 2.
        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with boxes
                or not. If True, the length of boxes and ``points`` should be
                the same. Defaults to False.
            eps (float): Make sure the points are inside not on the boundary.
                Defaults to 0.01.

        Returns:
            BoolTensor: A BoolTensor indicating whether a point is inside
            boxes. Assuming the boxes has shape of (n, 8), if ``is_aligned``
            is False. The index has shape of (m, n). If ``is_aligned`` is
            True, m should be equal to n and the index has shape of (m, ).
        """
        boxes = self.tensor
        assert boxes.dim() == 2, 'boxes dimension must be 2.'

        corners = boxes.reshape(-1, 4, 2)
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
            assert boxes.size(0) == points.size(0)

        values = (x1 - pt_x) * (y2 - pt_y) - (y1 - pt_y) * (x2 - pt_x)
        return (values >= eps).all(dim=-1) | (values <= -eps).all(dim=-1)

    @staticmethod
    def overlaps(boxes1: BaseBoxes,
                 boxes2: BaseBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two set of boxes with their modes
        converted to ``QuadriBoxes``.

        Args:
            boxes1 (:obj:`BaseBoxes`): BaseBoxes with shape of (m, box_dim)
                or empty.
            boxes2 (:obj:`BaseBoxes`): BaseBoxes with shape of (n, box_dim)
                or empty.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground). Defaults to "iou".
            is_aligned (bool): If True, then m and n must be equal. Defaults
                to False.
            eps (float): A value added to the denominator for numerical
                stability. Defaults to 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """
        raise NotImplementedError

    def from_instance_masks(masks: MaskType) -> 'QuadriBoxes':
        """Create boxes from instance masks.

        Args:
            masks (:obj:`BitmapMasks` or :obj:`PolygonMasks`): BitmapMasks or
                PolygonMasks instance with length of n.

        Returns:
            :obj:`QuadriBoxes`: Converted boxes with shape of (n, 8).
        """
        num_masks = len(masks)
        if num_masks == 0:
            return QuadriBoxes([], dtype=torch.float32)

        boxes = []
        if isinstance(masks, PolygonMasks):
            for idx, poly_per_obj in enumerate(masks.masks):
                pts_per_obj = []
                for p in poly_per_obj:
                    pts_per_obj.append(
                        np.array(p, dtype=np.float32).reshape(-1, 2))
                pts_per_obj = np.concatenate(pts_per_obj, axis=0)
                rect = cv2.minAreaRect(pts_per_obj)
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv2.boxPoints(rect)
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        else:
            masks = masks.to_ndarray()
            for idx in range(num_masks):
                coor_y, coor_x = np.nonzero(masks[idx])
                points = np.stack([coor_x, coor_y], axis=-1).astype(np.float32)
                rect = cv2.minAreaRect(points)
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv2.boxPoints(rect)
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        return QuadriBoxes(boxes)
