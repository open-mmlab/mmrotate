# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes, register_box
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torch import BoolTensor, Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]


@register_box('rbox')
class RotatedBoxes(BaseBoxes):
    """The rotated box class used in MMRotate by default.

    The ``box_dim`` of ``RotatedBoxes`` is 5, which means the length of the
    last dimension of the input should be 5. Each row of data means
    (x, y, w, h, t), where 'x' and 'y' are the coordinates of the box center,
    'w' and 'h' are the length of box sides, 't' is the box angle represented
    in radian. A rotated box can be regarded as rotating the horizontal box
    (x, y, w, h) w.r.t its center by 't' radian CW.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape
            (..., 5).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    box_dim = 5

    def regularize_boxes(self,
                         pattern: Optional[str] = None,
                         width_longer: bool = True,
                         start_angle: float = -90) -> Tensor:
        """Regularize rotated boxes.

        Due to the angle periodicity, one rotated box can be represented in
        many different (x, y, w, h, t). To make each rotated box unique,
        ``regularize_boxes`` will take the remainder of the angle divided by
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
        ``regualrize_boxes``:

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
        boxes = self.tensor
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

        x, y, w, h, t = boxes.unbind(dim=-1)
        if width_longer:
            # swap edge and angle if h >= w
            w_ = torch.where(w > h, w, h)
            h_ = torch.where(w > h, h, w)
            t = torch.where(w > h, t, t + np.pi / 2)
            t = ((t - start_angle) % np.pi) + start_angle
        else:
            # swap edge and angle if angle > pi/2
            t = ((t - start_angle) % np.pi)
            w_ = torch.where(t < np.pi / 2, w, h)
            h_ = torch.where(t < np.pi / 2, h, w)
            t = torch.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
        self.tensor = torch.stack([x, y, w_, h_, t], dim=-1)
        return self.tensor

    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes.

        If boxes have shape of (m, 8), centers have shape of (m, 2).
        """
        return self.tensor[..., :2]

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes.

        If boxes have shape of (m, 8), areas have shape of (m, ).
        """
        return self.tensor[..., 2] * self.tensor[..., 3]

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes.

        If boxes have shape of (m, 8), widths have shape of (m, ).
        """
        return self.tensor[..., 2]

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes.

        If boxes have shape of (m, 8), heights have shape of (m, ).
        """
        return self.tensor[..., 3]

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
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - flipped[..., 1]
            flipped[..., 4] = -flipped[..., 4]
        else:
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 1] = img_shape[0] - flipped[..., 1]

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        boxes = self.tensor
        assert len(distances) == 2
        boxes[..., :2] = boxes[..., :2] + boxes.new_tensor(distances)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        In ``RotatedBoxes``, ``clip`` function does nothing about the original
        data, because it's very tricky to handle rotate boxes corssing the
        image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        warnings.warn('The `clip` function does nothing in `RotatedBoxes`.')

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

        centers, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        t = t + angle / 180 * np.pi
        centers = torch.cat(
            [centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        centers_T = torch.transpose(centers, -1, -2)
        centers_T = torch.matmul(rotation_matrix, centers_T)
        centers = torch.transpose(centers_T, -1, -2)
        self.tensor = torch.cat([centers, wh, t], dim=-1)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        boxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)
        corners = self.rbox2corner(boxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = self.corner2rbox(corners)

    @staticmethod
    def rbox2corner(boxes: Tensor) -> Tensor:
        """Convert rotated box (x, y, w, h, t) to corners ((x1, y1), (x2, y1),
        (x1, y2), (x2, y2)).

        Args:
            boxes (Tensor): Rotated box tensor with shape of (..., 5).

        Returns:
            Tensor: Corner tensor with shape of (..., 4, 2).
        """
        ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
        cos_value, sin_value = torch.cos(theta), torch.sin(theta)
        vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
        vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return torch.stack([pt1, pt2, pt3, pt4], dim=-2)

    @staticmethod
    def corner2rbox(corners: Tensor) -> Tensor:
        """Convert corners ((x1, y1), (x2, y1), (x1, y2), (x2, y2)) to rotated
        box (x, y, w, h, t).

        Args:
            corners (Tensor): Corner tensor with shape of (..., 4, 2).

        Returns:
            Tensor: Rotated box tensor with shape of (..., 5).
        """
        original_shape = corners.shape[:-2]
        points = corners.cpu().numpy().reshape(-1, 4, 2)
        rboxes = []
        for pts in points:
            (x, y), (w, h), angle = cv2.minAreaRect(pts)
            rboxes.append([x, y, w, h, angle / 180 * np.pi])
        rboxes = corners.new_tensor(rboxes)
        return rboxes.reshape(*original_shape, 5)

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
        scale_x, scale_y = scale_factor
        ctrs, w, h, t = torch.split(boxes, [2, 1, 1, 1], dim=-1)
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
        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        scale_factor = boxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        self.tensor = torch.cat([ctrs, wh, t], dim=-1)

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
            the image. Assuming the original boxes have shape (m, n, 5),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        boxes = self.tensor
        return (boxes[..., 0] <= img_w + allowed_border) & \
               (boxes[..., 1] <= img_h + allowed_border) & \
               (boxes[..., 0] >= -allowed_border) & \
               (boxes[..., 1] >= -allowed_border)

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
            BoolTensor: A BoolTensor indicating whether the box is inside the
            image. Assuming the boxes has shape of (n, 5), if ``is_aligned``
            is False. The index has shape of (m, n). If ``is_aligned`` is True,
            m should be equal to n and the index has shape of (m, ).
        """
        boxes = self.tensor
        assert boxes.dim() == 2, 'boxes dimension must be 2.'

        if not is_aligned:
            boxes = boxes[None, :, :]
            points = points[:, None, :]
        else:
            assert boxes.size(0) == points.size(0)

        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)
        matrix = torch.cat([cos_value, sin_value, -sin_value, cos_value],
                           dim=-1).reshape(*boxes.shape[:-1], 2, 2)

        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        return (offset_x <= w / 2 - eps) & (offset_x >= - w / 2 + eps) & \
            (offset_y <= h / 2 - eps) & (offset_y >= - h / 2 + eps)

    @staticmethod
    def overlaps(boxes1: BaseBoxes,
                 boxes2: BaseBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two set of boxes with their types
        converted to ``RotatedBoxes``.

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
        from mmrotate.structures.bbox import rbbox_overlaps
        boxes1 = boxes1.convert_to('rbox')
        boxes2 = boxes2.convert_to('rbox')
        return rbbox_overlaps(
            boxes1.tensor,
            boxes2.tensor,
            mode=mode,
            is_aligned=is_aligned,
            eps=eps)

    @staticmethod
    def from_instance_masks(masks: MaskType) -> 'RotatedBoxes':
        """Create boxes from instance masks.

        Args:
            masks (:obj:`BitmapMasks` or :obj:`PolygonMasks`): BitmapMasks or
                PolygonMasks instance with length of n.

        Returns:
            :obj:`RotatedBoxes`: Converted boxes with shape of (n, 5).
        """
        num_masks = len(masks)
        if num_masks == 0:
            return RotatedBoxes([], dtype=torch.float32)

        boxes = []
        if isinstance(masks, BitmapMasks):
            for idx in range(num_masks):
                mask = masks.masks[idx]
                points = np.stack(np.nonzero(mask), axis=-1).astype(np.float32)
                (x, y), (w, h), angle = cv2.minAreaRect(points)
                boxes.append([x, y, w, h, angle / 180 * np.pi])
        elif isinstance(masks, PolygonMasks):
            for idx, poly_per_obj in enumerate(masks.masks):
                pts_per_obj = []
                for p in poly_per_obj:
                    pts_per_obj.append(
                        np.array(p, dtype=np.float32).reshape(-1, 2))
                pts_per_obj = np.concatenate(pts_per_obj, axis=0)
                (x, y), (w, h), angle = cv2.minAreaRect(pts_per_obj)
                boxes.append([x, y, w, h, angle / 180 * np.pi])
        else:
            raise TypeError(
                '`masks` must be `BitmapMasks`  or `PolygonMasks`, '
                f'but got {type(masks)}.')
        return RotatedBoxes(boxes)
