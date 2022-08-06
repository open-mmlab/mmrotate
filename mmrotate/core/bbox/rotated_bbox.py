# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes, register_bbox_mode
from torch import Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]


@register_bbox_mode('rbbox')
class RotatedBoxes(BaseBoxes):

    _bbox_dim = 5

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

    def regular_bboxes(self, pattern=None, width_longer=True, start_angle=-90):
        bboxes = self.tensor
        pattern_settings = dict(
            oc=(False, -90), le90=(True, -90), le135=(True, -45))
        if pattern is not None:
            assert pattern in ['oc', 'le90', 'le135']
            width_longer, start_angle = pattern_settings[pattern]
        start_angle = start_angle / 180. * np.pi

        x, y, w, h, t = bboxes.unbind(dim=-1)
        if width_longer:
            w_ = torch.where(w > h, w, h)
            h_ = torch.where(w > h, h, w)
            t = torch.where(w > h, t, t + np.pi / 2)
            t = ((t - start_angle) % np.pi) + start_angle
        else:
            t = ((t - start_angle) % np.pi)
            w_ = torch.where(t < np.pi / 2, w, h)
            h_ = torch.where(t < np.pi / 2, h, w)
            t = torch.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
        bboxes = torch.stack([x, y, w_, h_, t], dim=-1)
        return bboxes

    def flip(self: T,
             img_shape: Tuple[int, int],
             direction: str = 'horizontal') -> T:
        """Flip bboxes horizontally or vertically.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"

        Returns:
            T: Flipped boxes with the same shape as the original boxes.
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        bboxes = self.tensor
        flipped = bboxes.clone()
        if direction == 'horizontal':
            flipped[..., 0] = img_shape[1] - bboxes[..., 0]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - bboxes[..., 1]
        else:
            flipped[..., 0] = img_shape[1] - bboxes[..., 0]
            flipped[..., 1] = img_shape[0] - bboxes[..., 1]
        flipped[..., 4] = -flipped[..., 4]
        return type(self)(flipped)

    def translate(self: T, distances: Tuple[float, float]) -> T:
        """Translate bboxes.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.

        Returns:
            T: Translated boxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        assert len(distances) == 2
        bboxes[..., :2] = bboxes[..., :2] + bboxes.new_tensor(distances)
        return type(self)(bboxes)

    def clip(self: T, img_shape: Tuple[int, int]) -> T:
        """Clip boxes according to border.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            T: Cliped boxes with the same shape as the original boxes.
        """
        return type(self)(self.tensor.clone())

    def rotate(self: T,
               center: Tuple[float, float],
               angle: float,
               img_shape: Optional[Tuple[int, int]] = None) -> T:
        """Rotate all boxes.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle.
            img_shape (Tuple[int, int], Optional): A tuple of image height
                and width. Defaults to None.

        Returns:
            T: Rotated boxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        rotation_matrix = bboxes.new_tensor(
            cv2.getRotationMatrix2D(center, angle, 1))

        centers, wh, t = torch.split(bboxes, [2, 2, 1], dim=-1)
        t = t - angle / 180 * np.pi
        centers = torch.cat(
            [centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        centers_T = torch.transpose(centers, -1, -2)
        centers_T = torch.matmul(rotation_matrix, centers_T)
        centers = torch.transpose(centers_T, -1, -2)
        bboxes = torch.cat([centers, wh, t], dim=-1)
        return type(self)(bboxes)

    def project(self: T,
                homography_matrix: Union[Tensor, np.ndarray],
                img_shape: Optional[Tuple[int, int]] = None) -> T:
        """Geometric transformation for bbox.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
            img_shape (Tuple[int, int], Optional): A tuple of image height
                and width. Defaults to None.

        Returns:
            T: Converted bboxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        corners = self.rbbox2corner(bboxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        return self(type)(self.corner2rbbox(corners))

    @staticmethod
    def rbbox2corner(bboxes):
        ctr, w, h, theta = torch.split(bboxes, (2, 1, 1, 1), dim=-1)
        Cos, Sin = torch.cos(theta), torch.sin(theta)
        vec1 = torch.cat([w / 2 * Cos, w / 2 * Sin], dim=-1)
        vec2 = torch.cat([-h / 2 * Sin, h / 2 * Cos], dim=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return torch.stack([pt1, pt2, pt3, pt4], dim=-2)

    @staticmethod
    def corner2rbbox(corners):
        original_shape = corners.shape[:-2]
        points = corners.cpu().numpy().reshape(-1, 4, 2)
        rbboxes = []
        for pts in points:
            (x, y), (w, h), angle = cv2.minAreaRect(pts)
            rbboxes.append([x, y, w, h, angle / 180 * np.pi])
        rbboxes = corners.new_tensor(rbboxes)
        return rbboxes.view(*original_shape, 5)

    def rescale(self: T,
                scale_factor: Tuple[float, float],
                mapping_back=False) -> T:
        """Rescale boxes w.r.t. rescale_factor.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
            mapping_back (bool): Mapping back the rescaled bboxes.
                Defaults to False.

        Returns:
            T: Rescaled boxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        assert len(scale_factor) == 2
        scale_x, scale_y = scale_factor
        ctrs, w, h, t = torch.split(bboxes, [2, 1, 1, 1], dim=-1)
        Cos, Sin = torch.cos(t), torch.Sin(t)

        if mapping_back:
            scale_x, scale_y = 1 / scale_x, 1 / scale_y

        # Refer to https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/rotated_boxes.py # noqa
        # rescale centers
        ctrs = ctrs * ctrs.new_tensor([scale_x, scale_y])
        # rescale width and height
        w = w * torch.sqrt((scale_x * Cos)**2 + (scale_y * Sin)**2)
        h = h * torch.sqrt((scale_x * Sin)**2 + (scale_y * Cos)**2)
        # recalculate theta
        t = torch.atan2(scale_x * Sin, scale_y * Cos)

        bboxes = torch.cat([ctrs, w, h, t], dim=-1)
        return type(self)(bboxes)

    def resize_bboxes(self: T, scale_factor: Tuple[float, float]) -> T:
        """Resize the box width and height w.r.t scale_factor.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.

        Returns:
            Tensor: Resized bboxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        assert len(scale_factor) == 2
        ctrs, wh, t = torch.split(bboxes, [2, 2, 1], dim=-1)
        scale_factor = bboxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        bboxes = torch.cat([ctrs, wh, t], dim=-1)
        return type(self)(bboxes)

    def is_bboxes_inside(self, img_shape: Tuple[int, int]) -> torch.BoolTensor:
        """Find bboxes as long as a part of bboxes is inside an region.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            BoolTensor: Index of the remaining bboxes. Assuming the original
            boxes have shape (a0, a1, ..., bbox_dim), the output has shape
            (a0, a1, ...).
        """
        img_h, img_w = img_shape
        bboxes = self.tensor
        return (bboxes[..., 0] < img_w) & (bboxes[..., 0] > 0) \
            & (bboxes[..., 1] < img_h) & (bboxes[..., 1] > 0)

    def find_inside_points(self, points: Tensor) -> torch.BoolTensor:
        """Find inside box points.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).

        Returns:
            BoolTensor: Index of inside box points. Has shape of (m, n)
                where n is the length of the boxes after flattening.
        """
        bboxes = self.tensor
        if bboxes.dim() > 2:
            bboxes = bboxes.flatten(end_dim=-2)

        bboxes = bboxes[None, :, :]
        points = points[:, None, :]
        num_points, num_bboxes = points.size(0), bboxes.size(0)
        ctrs, wh, t = torch.split(bboxes, [2, 2, 1], dim=2)
        Cos, Sin = torch.cos(t), torch.sin(t)
        matrix = torch.cat([Cos, Sin, -Sin, Cos],
                           dim=-1).view(num_points, num_bboxes, 2, 2)

        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        return (offset_x < w / 2) & (offset_x > - w / 2) & \
            (offset_y < h / 2) & (offset_y > - h / 2)
