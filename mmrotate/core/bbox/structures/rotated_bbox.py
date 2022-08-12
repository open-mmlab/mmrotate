# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes, register_bbox_mode
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torch import BoolTensor, Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]


@register_bbox_mode('rbox')
class RotatedBoxes(BaseBoxes):
    """The rotated box class used in MMRotate by default.

    The ``bbox_dim`` of ``RotatedBoxes`` is 5, which means the length of the
    last dimension of the input should be 5. Each row of data means
    (x, y, w, h, t), where 'x' and 'y' are the coordinates of the box center,
    'w' and 'h' are the length of box sides, 't' is the box angle represented
    in radian. A rotated box can be regarded as rotating the horizontal box
    (x, y, w, h) w.r.t its center by 't' radian CW.

    Args:
        bboxes (Tensor or np.ndarray or Sequence): The box data with
            shape (..., 5).
        dtype (torch.dtype, Optional): data type of bboxes. Defaults to None.
        device (str or torch.device, Optional): device of bboxes.
            Default to None.
        clone (bool): Whether clone ``bboxes`` or not. Defaults to True.
    """

    bbox_dim = 5

    def regularize_bboxes(self,
                          pattern: Optional[str] = None,
                          width_longer: bool = True,
                          start_angle: float = -90) -> Tensor:
        """Regularize rotated boxes.

        Due to the angle periodicity, one rotated box can be represented in
        many different (x, y, w, h, t). To make each rotated box unique,
        ``regularize_bboxes`` will take the remainder of the angle divided by
        180 degrees.

        However, after taking the remainder of the angle, there are still two
        representations for one rotate box. For example, (0, 0, 4, 5, 0.5) and
        (0, 0, 5, 4, 0.5 + pi/2) are the same areas in the image. To solve the
        problem, the code will swap edges w.r.t ``width_longer``:

        - width_longer=True: Make sure the width is longer than the height. If
          not, swap the width and height. The angle ranges in [start_angle,
          start_angle + 180). For the above example, the rotated box will be
          represented as (0, 0, 5, 4, 0.5 + pi/2).
        - width_longer=False: Make sure the angle is lower than
          start_angle+pi/2. If not, swap the width and height. The angle
          ranges in [start_angle, start_angle + 90). For the above example,
          the rotated box will be represented as (0, 0, 4, 5, 0.5).

        For convenience, three commonly used patterns are preset in
        ``regualrize_bboxes``:

        - 'oc': OpenCV Definition. Has the same box representation as
          ``cv2.minAreaRect`` the angle ranges in [-90, 0). Equal to set
          width_longer=False and start_angle=-90.
        - 'le90': Long Edge Definition (90). the angle ranges in [-90, 90).
          The width is always longer than the height. Equal to set
          width_longer=True and start_angle=-90.
        - 'le135': Long Edge Definition (135). the angle ranges in [-45, 135).
          The width is always longer than the height. Equal to set
          width_longer=True and start_angle=-45.

        Args:
            pattern (str, Optional): Regularization pattern. Can only be 'oc',
                'le90', or 'le135'. Defaults to None.
            width_longer (bool): Whether to make sure width is larger than
                height. Defaults to True.
            start_angle (float): The starting angle of the box angle
                represented in degrees. Defaults to -90.

        Returns:
            Tensor: Regularized box tensor.
        """
        bboxes = self.tensor
        if pattern is not None:
            if pattern == 'oc':
                width_longer, start_angle = False, -90
            elif pattern == 'le90':
                width_longer, start_angle = True, -90
            elif pattern == 'le135':
                width_longer, start_angle = True, -45
            else:
                raise ValueError("pattern only can be 'oc', 'le90', and"
                                 f"'le135', but get {pattern}.")
        start_angle = start_angle / 180 * np.pi

        x, y, w, h, t = bboxes.unbind(dim=-1)
        if width_longer:
            # swap edge and angle if h >= w
            w_ = torch.where(w > h, w, h)
            h_ = torch.where(w > h, h, w)
            t = torch.where(w > h, t, t + np.pi / 2)
            t = ((t - start_angle) % np.pi) + start_angle
        else:
            # swap edge and angle if angle >= pi/2
            t = ((t - start_angle) % np.pi)
            w_ = torch.where(t < np.pi / 2, w, h)
            h_ = torch.where(t < np.pi / 2, h, w)
            t = torch.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
        bboxes = torch.stack([x, y, w_, h_, t], dim=-1)
        return bboxes

    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes."""
        return self.tensor[..., :2]

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes."""
        return self.tensor[..., 2] * self.tensor[..., 3]

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes."""
        return self.tensor[..., 2]

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes."""
        return self.tensor[..., 3]

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
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - flipped[..., 1]
            flipped[..., 4] = -flipped[..., 4]
        else:
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 1] = img_shape[0] - flipped[..., 1]

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Inplace translate bboxes.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        bboxes = self.tensor
        assert len(distances) == 2
        bboxes[..., :2] = bboxes[..., :2] + bboxes.new_tensor(distances)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Inplace clip boxes according to the image shape.

        In ``RotatedBoxes``, ``clip`` function does nothing about the original
        data, because it's very tricky to handle rotate boxes corssing the
        image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        warnings.warn('The `clip` function does nothing in `RotatedBoxes`.')

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Inplace rotate all boxes.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        bboxes = self.tensor
        rotation_matrix = bboxes.new_tensor(
            cv2.getRotationMatrix2D(center, -angle, 1))

        centers, wh, t = torch.split(bboxes, [2, 2, 1], dim=-1)
        t = t + angle / 180 * np.pi
        centers = torch.cat(
            [centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        centers_T = torch.transpose(centers, -1, -2)
        centers_T = torch.matmul(rotation_matrix, centers_T)
        centers = torch.transpose(centers_T, -1, -2)
        self.tensor = torch.cat([centers, wh, t], dim=-1)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Inplace geometric transformation for bbox.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        bboxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = bboxes.new_tensor(homography_matrix)
        corners = self.rbbox2corner(bboxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = self.corner2rbbox(corners)

    @staticmethod
    def rbbox2corner(bboxes: Tensor) -> Tensor:
        """Convert rotated bbox (x, y, w, h, t) to corners ((x1, y1), (x2, y1),
        (x1, y2), (x2, y2)).

        Args:
            bboxes (Tensor): Rotated box tensor with shape of (..., 5).

        Returns:
            Tensor: Corner tensor with shape of (..., 4, 2).
        """
        ctr, w, h, theta = torch.split(bboxes, (2, 1, 1, 1), dim=-1)
        cos_value, sin_value = torch.cos(theta), torch.sin(theta)
        vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
        vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return torch.stack([pt1, pt2, pt3, pt4], dim=-2)

    @staticmethod
    def corner2rbbox(corners: Tensor) -> Tensor:
        """Convert corners ((x1, y1), (x2, y1), (x1, y2), (x2, y2)) to rotated
        box (x, y, w, h, t).

        Args:
            corners (Tensor): Corner tensor with shape of (..., 4, 2).

        Returns:
            Tensor: Rotated box tensor with shape of (..., 5).
        """
        original_shape = corners.shape[:-2]
        points = corners.cpu().numpy().reshape(-1, 4, 2)
        rbboxes = []
        for pts in points:
            (x, y), (w, h), angle = cv2.minAreaRect(pts)
            rbboxes.append([x, y, w, h, angle / 180 * np.pi])
        rbboxes = corners.new_tensor(rbboxes)
        return rbboxes.reshape(*original_shape, 5)

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Inplace rescale boxes w.r.t. rescale_factor.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink bboxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of bboxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        bboxes = self.tensor
        assert len(scale_factor) == 2
        scale_x, scale_y = scale_factor
        ctrs, w, h, t = torch.split(bboxes, [2, 1, 1, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)

        # Refer to https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/rotated_boxes.py # noqa
        # rescale centers
        ctrs = ctrs * ctrs.new_tensor([scale_x, scale_y])
        # rescale width and height
        w = w * torch.sqrt((scale_x * cos_value)**2 + (scale_y * sin_value)**2)
        h = h * torch.sqrt((scale_x * sin_value)**2 + (scale_y * cos_value)**2)
        # recalculate theta
        t = torch.atan2(scale_x * sin_value, scale_y * cos_value)
        self.tensor = torch.cat([ctrs, w, h, t], dim=-1)

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
        ctrs, wh, t = torch.split(bboxes, [2, 2, 1], dim=-1)
        scale_factor = bboxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        self.tensor = torch.cat([ctrs, wh, t], dim=-1)

    def is_bboxes_inside(self, img_shape: Tuple[int, int]) -> BoolTensor:
        """Find bboxes inside the image.

        In ``RotatedBoxes``, as long as the center of box is inside the image,
        this box will be regarded as True.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 5),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        bboxes = self.tensor
        return (bboxes[..., 0] <= img_w) & (bboxes[..., 0] >= 0) \
            & (bboxes[..., 1] <= img_h) & (bboxes[..., 1] >= 0)

    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False) -> BoolTensor:
        """Find inside box points. Boxes dimension must be 2.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with bboxes
                or not. If True, the length of bboxes and ``points`` should be
                the same. Defaults to False.

        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside the
            image. Assuming the boxes has shape of (n, 5), if ``is_aligned``
            is False. The index has shape of (m, n). If ``is_aligned`` is True,
            m should be equal to n and the index has shape of (m, ).
        """
        bboxes = self.tensor
        assert bboxes.dim() == 2, 'bboxes dimension must be 2.'

        if not is_aligned:
            bboxes = bboxes[None, :, :]
            points = points[:, None, :]
        else:
            assert bboxes.size(0) == points.size(0)

        ctrs, wh, t = torch.split(bboxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)
        matrix = torch.cat([cos_value, sin_value, -sin_value, cos_value],
                           dim=-1).reshape(*bboxes.shape[:-1], 2, 2)

        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        return (offset_x <= w / 2) & (offset_x >= - w / 2) & \
            (offset_y <= h / 2) & (offset_y >= - h / 2)

    @staticmethod
    def bbox_overlaps(bboxes1: BaseBoxes,
                      bboxes2: BaseBoxes,
                      mode: str = 'iou',
                      is_aligned: bool = False,
                      eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two set of boxes with their modes
        converted to ``RotatedBoxes``.

        Args:
            bboxes1 (:obj:`BaseBoxes`): BaseBoxes with shape of (m, bbox_dim)
                or empty.
            bboxes2 (:obj:`BaseBoxes`): BaseBoxes with shape of (n, bbox_dim)
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
        from ..iou_calculators import rbbox_overlaps
        bboxes1 = bboxes1.convert_to('rbox')
        bboxes2 = bboxes2.convert_to('rbox')
        return rbbox_overlaps(
            bboxes1.tensor,
            bboxes2.tensor,
            mode=mode,
            is_aligned=is_aligned,
            eps=eps)

    @staticmethod
    def from_bitmap_masks(masks: BitmapMasks) -> 'RotatedBoxes':
        """Create boxes from ``BitmapMasks``.

        Args:
            masks (:obj:`BitmapMasks`): BitmapMasks with length of n.

        Returns:
            :obj:`RotatedBoxes`: Converted boxes with shape of (n, 5).
        """
        num_masks = len(masks)
        if num_masks == 0:
            return RotatedBoxes(np.zeros((0, 5), dtype=np.float32))
        boxes = []
        for idx in range(num_masks):
            mask = masks.masks[idx]
            points = np.stack(np.nonzero(mask), axis=-1).astype(np.float32)
            (x, y), (w, h), angle = cv2.minAreaRect(points)
            boxes.append([x, y, w, h, angle / 180 * np.pi])
        return RotatedBoxes(boxes)

    @staticmethod
    def from_polygon_masks(masks: PolygonMasks) -> 'RotatedBoxes':
        """Create boxes from ``PolygonMasks``.

        Args:
            masks (:obj:`BitmapMasks`): PolygonMasks

        Returns:
            :obj:`RotatedBoxes`: Converted boxes with shape of (n, 5).
        """
        num_masks = len(masks)
        if num_masks == 0:
            return RotatedBoxes(np.zeros((0, 5), dtype=np.float32))
        boxes = []
        for idx, poly_per_obj in enumerate(masks.masks):
            pts_per_obj = []
            for p in poly_per_obj:
                pts_per_obj.append(
                    np.array(p, dtype=np.float32).reshape(-1, 2))
            pts_per_obj = np.concatenate(pts_per_obj, axis=0)
            (x, y), (w, h), angle = cv2.minAreaRect(pts_per_obj)
            boxes.append([x, y, w, h, angle / 180 * np.pi])
        return RotatedBoxes(boxes)
