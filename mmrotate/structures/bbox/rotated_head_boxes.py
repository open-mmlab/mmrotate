# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import register_box
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torch import BoolTensor, Tensor

from .rotated_boxes import RotatedBoxes

T = TypeVar('T')
DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]


@register_box('rheadbox')
class RotatedHeadBoxes(RotatedBoxes):
    """The rotated head box class used in MMRotate.

    The 'RotatedHeadBoxes' add the head of rotated bboxes. You can get the head
    of rotated bboxes by property 'head', and all the other functions and
    properties are exactly the same in RotatedBoxes.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape
            (..., 10). The
            the head quadrant of rotated bboxes.
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    box_dim = 7

    def __init__(self,
                 data: Union[Tensor, np.ndarray],
                 dtype: torch.dtype = None,
                 device: DeviceType = None,
                 clone: bool = True) -> None:
        if isinstance(data, (np.ndarray, Tensor, Sequence)):
            data = torch.as_tensor(data)
        else:
            raise TypeError('boxes should be Tensor, ndarray, or Sequence, ',
                            f'but got {type(data)}')
        if device is not None or dtype is not None:
            data = data.to(dtype=dtype, device=device)
        # Clone the data to avoid potential bugs
        if clone:
            data = data.clone()
        # handle the empty input like []
        if data.numel() == 0:
            data = data.reshape((-1, self.box_dim))
        assert data.dim() >= 2 and (data.size(-1) == self.box_dim or
                                    data.size(-1) == 10),\
            ('The boxes dimension must >= 2 and the length of the last '
             f'dimension must be {self.box_dim}, but got boxes with '
             f'shape {data.shape}.')
        if data.size(-1) == 10:
            self.tensor = self.get_head_and_rbox(data[..., :-2], data[...,
                                                                      -2:])
        elif data.size(-1) == self.box_dim:
            self.tensor = data
        else:
            raise ValueError(
                'The length of last dim of data is not correct, }'
                f'expect {self.box_dim}, but got data with shape {data.shape}.'
            )

    def get_head_and_rbox(self, qbox_data: Tensor, head_data: Tensor):
        """get head quadrant and rbox params from qbox data and head data.

        Args:
            qbox_data (Tensor): qbox format data, shape(..., 8).
            head_data (Tensor): head xy data, shape(..., 2).
        """
        assert qbox_data.size(-1) == 8 and head_data.size(-1) == 2, \
            ('The last dimension of two input params is incorrect, expect 2 '
             'or 8, '
             f'but got qbox_data {qbox_data.shape}, head_data '
             f'{head_data.shape}.')
        original_shape = qbox_data.shape[:-1]
        points = qbox_data.cpu().numpy().reshape(-1, 4, 2)
        rboxes = []
        for pts in points:
            (x, y), (w, h), angle = cv2.minAreaRect(pts)
            rboxes.append([x, y, w, h, angle / 180 * np.pi])
        rboxes = qbox_data.new_tensor(rboxes)
        rboxes = rboxes.view(*original_shape, 5)
        return torch.cat([rboxes, head_data], dim=-1)

    def get_head_quadrant(self, center_xys: Tensor, head_xys: Tensor):
        """get head quadrant from head xy.

        Args:
            center_xys (Tensor): bboxes center coordinates, shape (..., 2)
            head_xys (Tensor): heads xy coordinates, shape (m, 2)
        """
        center_xys = center_xys.reshape(-1, 2)
        original_shape = head_xys.shape[:-1]
        assert center_xys.size(-1) == 2 and head_xys.size(-1) == 2, \
            ('The last dimension of two input params must be 2, representing '
             f'xy coordinates, but got center_xys {center_xys.shape}, '
             f'head_xys {head_xys.shape}.')
        head_quadrants = []
        for center_xy, head_xy in zip(center_xys, head_xys):
            delta_x = head_xy[0] - center_xy[0]
            delta_y = head_xy[1] - center_xy[1]
            if (delta_x >= 0) and (delta_y >= 0):
                head_quadrants.append(0)
            elif (delta_x >= 0) and (delta_y <= 0):
                head_quadrants.append(1)
            elif (delta_x <= 0) and (delta_y <= 0):
                head_quadrants.append(2)
            else:
                head_quadrants.append(3)
        head_quadrants = head_xys.new_tensor(head_quadrants)
        head_quadrants = head_quadrants.view(*original_shape)
        return head_quadrants

    def regularize_boxes(self,
                         pattern: Optional[str] = None,
                         width_longer: bool = True,
                         start_angle: float = -90) -> Tensor:
        """Regularize rotated boxes according to angle pattern and start angle.

        Args:
            pattern (str, Optional): Regularization pattern. Can only be 'oc',
                'le90', or 'le135'. Defaults to None.
            width_longer (bool): Whether to make sure width is larger than
                height. Defaults to True.
            start_angle (float): The starting angle of the box angle
                represented in degrees. Defaults to -90.

        Returns:
            Tensor: Regularized box tensor
        """
        boxes, headxs, headys = self.tensor[..., :5], self.tensor[
            ..., -2], self.tensor[..., -1]
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
        self.tensor = torch.stack([x, y, w_, h_, t, headxs, headys], dim=-1)
        return self.tensor[..., :5]

    @property
    def heads(self) -> Tensor:
        """Return a tensor representing the heads quadrant of boxes.

        if boxes have shape of (m, 8) or (m, 5), heads quadrant have shape of
        (m,)
        """
        center_xys, head_xys = self.tensor[..., :2], self.tensor[..., -2:]
        assert center_xys.size(-1) == 2 and head_xys.size(-1) == 2, \
            ('The last dimension of two input params must be 2, representing '
             f'xy coordinates, but got center_xys {center_xys.shape}, '
             f'head_xys {head_xys.shape}.')
        head_quadrants = self.get_head_quadrant(center_xys, head_xys)
        return head_quadrants.int()

    @property
    def head_xys(self) -> Tensor:
        """Return a tensor representing the heads xy of boxes.

        if boxes have shape of (m, 8) or (m, 5), head_xys have shape of (m, 2)
        """
        head_xys = self.tensor[..., -2:]
        return head_xys

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
            flipped[..., -2] = img_shape[1] - flipped[..., -2]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - flipped[..., 1]
            flipped[..., -1] = img_shape[0] - flipped[..., -1]
            flipped[..., 4] = -flipped[..., 4]
        else:
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., -2] = img_shape[1] - flipped[..., -2]
            flipped[..., 1] = img_shape[0] - flipped[..., 1]
            flipped[..., -1] = img_shape[0] - flipped[..., -1]

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degree. Positive
                values mean clockwise rotation.
        """
        boxes = self.tensor
        rotation_matrix = boxes.new_tensor(
            cv2.getRotationMatrix2D(center, -angle, 1))
        centers, wh, t, head_xys = torch.split(boxes, [2, 2, 1, 2], dim=-1)
        t = t + angle / 180 * np.pi
        centers = torch.cat(
            [centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        head_xys = torch.cat(
            [head_xys, head_xys.new_ones(*head_xys.shape[:-1], 1)], dim=-1)
        centers_T = torch.transpose(centers, -1, -2)
        centers_T = torch.matmul(rotation_matrix, centers_T)
        head_xys_T = torch.transpose(head_xys, -1, -2)
        head_xys_T = torch.matmul(rotation_matrix, head_xys_T)
        centers = torch.transpose(centers_T, -1, -2)
        head_xys = torch.transpose(head_xys_T, -1, -2)
        self.tensor = torch.cat([centers, wh, t, head_xys], dim=-1)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        boxes, head_xys = self.tensor[..., :5], self.tensor[..., 5:]
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)
        corners = self.rbox2corner(boxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        head_xys = torch.cat(
            [head_xys, head_xys.new_ones(*head_xys.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        head_xys_T = torch.transpose(head_xys, -1, -2)
        head_xys_T = torch.matmul(homography_matrix, head_xys_T)
        corners = torch.transpose(corners_T, -1, -2)
        head_xys = torch.transpose(head_xys_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        head_xys = head_xys[..., :2] / head_xys[..., 2:3]
        boxes = self.corner2rbox(corners)
        self.tensor = torch.cat([boxes, head_xys], dim=-1)

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
        boxes, head_xys = self.tensor[..., :5], self.tensor[..., 5:]
        assert len(scale_factor) == 2
        scale_x, scale_y = scale_factor
        ctrs, w, h, t = torch.split(boxes, [2, 1, 1, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)

        # Refer to https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/rotated_boxes.py # noqa
        # rescale centers and head_xys
        ctrs = ctrs * ctrs.new_tensor([scale_x, scale_y])
        head_xys = head_xys * head_xys.new_tensor([scale_x, scale_y])
        # rescale width and height
        w = w * torch.sqrt((scale_x * cos_value)**2 + (scale_y * sin_value)**2)
        h = h * torch.sqrt((scale_x * sin_value)**2 + (scale_y * cos_value)**2)
        # recalculate theta
        t = torch.atan2(scale_x * sin_value, scale_y * cos_value)
        self.tensor = torch.cat([ctrs, w, h, t, head_xys], dim=-1)

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
        boxes, head_xys = self.tensor[..., :5], self.tensor[..., 5:]
        assert len(scale_factor) == 2
        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        scale_factor = boxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        self.tensor = torch.cat([ctrs, wh, t, head_xys], dim=-1)

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
        boxes = self.tensor[..., :5]
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
