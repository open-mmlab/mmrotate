"""copy from /mmrotate/structures/bbox/rotated_boxes.py and redefine 'r360'."""
import math
from typing import Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import register_box, register_box_converter
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torch import Tensor

from mmrotate.structures.bbox import QuadriBoxes
from mmrotate.structures.bbox import RotatedBoxes as mmrotate_RotatedBoxes

# from skimage.draw import line as skidline

T = TypeVar('T')
DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]
#  get_line = lambda start1, start2: list(
#  zip(*skidline(start1[0], start1[1], start2[0], start2[1])))


@register_box('rbox', force=True)
class RotatedBoxes(mmrotate_RotatedBoxes):
    """Copy form mmrotate/structures/bbox/RotatedBoxes add new rotate logic.

    +-180.

        -90
          |
          |
    +-180---------0
          |
          |
        +90
    [-180,180)
    """
    box_dim = 5

    def regularize_boxes(self,
                         pattern: Optional[str] = None,
                         width_longer: bool = True,
                         start_angle: float = -90) -> Tensor:
        boxes = self.tensor  # type: ignore
        if pattern is not None:
            if pattern == 'oc':
                width_longer, start_angle = False, -90
            elif pattern == 'le90':
                width_longer, start_angle = True, -90
            elif pattern == 'le135':
                width_longer, start_angle = True, -45
            elif pattern == 'r360':
                width_longer, start_angle = False, -180
            else:
                raise ValueError("pattern only can be 'oc', 'le90', 'r360' and"
                                 f"'le135', but get {pattern}.")
        start_angle = start_angle / 180 * np.pi

        x, y, w, h, t = boxes.unbind(dim=-1)
        if width_longer and pattern != 'r360':
            # swap edge and angle if h >= w
            w_ = torch.where(w > h, w, h)
            h_ = torch.where(w > h, h, w)
            t = torch.where(w > h, t, t + np.pi / 2)
            t = ((t - start_angle) % np.pi) + start_angle
        elif pattern != 'r360':
            t = ((t - start_angle) % np.pi)
            w_ = torch.where(t < np.pi / 2, w, h)
            h_ = torch.where(t < np.pi / 2, h, w)
            t = torch.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
        else:
            w_ = w
            h_ = h
            t = t % (2 * np.pi)
            t = torch.where(t >= np.pi, t - 2 * np.pi, t)
        self.tensor = torch.stack([x, y, w_, h_, t], dim=-1)
        return self.tensor

    @staticmethod
    def corner2rbox(corners: Tensor) -> Tensor:

        def dist_torch(pt1, pt2):
            return torch.norm(pt1 - pt2, dim=-1)

        original_shape = corners.shape[:-2]
        points = corners.reshape(-1, 4, 2)
        cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
        cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
        _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
        _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
        _thetas = torch.unsqueeze(
            torch.atan2((points[:, 1, 0] - points[:, 2, 0]),
                        (points[:, 1, 1] - points[:, 2, 1])),
            axis=1)
        thetas = torch.where(_thetas >= 0, math.pi - _thetas,
                             -(math.pi + _thetas))
        rbboxes = torch.cat([cxs, cys, _ws, _hs, thetas], axis=1)
        return rbboxes.reshape(*original_shape, 5)

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        diff with mmrotate first we normalize angle range[0,2*np.pi) after
        rotate we fix angle range to [-np.pi,np.pi)
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        flipped = self.tensor  # type: ignore
        flipped[..., 4] = flipped[..., 4] % (2 * np.pi)
        if direction == 'horizontal':
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 4] = (2 * np.pi) - flipped[..., 4]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - flipped[..., 1]
            flipped[..., 4] = (np.pi - flipped[..., 4]) % (2 * np.pi)
        elif direction == 'diagonal':
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 1] = img_shape[0] - flipped[..., 1]
            flipped[..., 4] = (flipped[..., 4] - np.pi) % (2 * np.pi)
        flipped[..., 4] = torch.where(flipped[..., 4] >= np.pi,
                                      flipped[..., 4] - 2 * np.pi, flipped[...,
                                                                           4])

    def rotate_auto_bound_(self, center: Tuple[float, float], angle: float,
                           img_shape_record) -> None:
        """Rotate all boxes in-place.

        copy from mmrotate
        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        boxes = self.tensor

        h, w, new_h, new_w = img_shape_record[0][0], img_shape_record[0][
            1], img_shape_record[1][0], img_shape_record[1][1]

        rotation_matrix = boxes.new_tensor(
            cv2.getRotationMatrix2D(((w - 1) * 0.5, (h - 1) * 0.5), -angle,
                                    1.0))

        # follow change the rotation matrix according to the change img shape
        # cos = np.abs(rotation_matrix[0, 0])
        # sin = np.abs(rotation_matrix[0, 1])
        # new_w = h * sin + w * cos
        # new_h = h * cos + w * sin

        rotation_matrix[0, 2] += (new_w - w) * 0.5
        rotation_matrix[1, 2] += (new_h - h) * 0.5

        # w = int(np.round(new_w))
        # h = int(np.round(new_h))

        #

        centers, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        t = t % (2 * np.pi) + angle / 180 * np.pi
        t = torch.where(t >= np.pi, t - 2 * np.pi, t)
        centers = torch.cat(
            [centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        centers_T = torch.transpose(centers, -1, -2)
        centers_T = torch.matmul(rotation_matrix, centers_T)
        centers = torch.transpose(centers_T, -1, -2)
        self.tensor = torch.cat([centers, wh, t], dim=-1)

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        copy from mmrotate
        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        boxes = self.tensor
        rotation_matrix = boxes.new_tensor(
            cv2.getRotationMatrix2D(center, -angle, 1))

        centers, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        t = t % (2 * np.pi) + angle / 180 * np.pi
        t = torch.where(t >= np.pi, t - 2 * np.pi, t)
        centers = torch.cat(
            [centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        centers_T = torch.transpose(centers, -1, -2)
        centers_T = torch.matmul(rotation_matrix, centers_T)
        centers = torch.transpose(centers_T, -1, -2)
        self.tensor = torch.cat([centers, wh, t], dim=-1)


@register_box_converter(RotatedBoxes, QuadriBoxes, force=True)
def rbox2qbox(boxes: Tensor) -> Tensor:
    """copy from mmrotate/structures/bbox/box_converters.py Convert rotated
    boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).
    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    centerx, centery, w, h, theta = torch.split(boxes, (1, 1, 1, 1, 1), dim=-1)
    # cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    cosa = torch.cos(theta)
    sina = torch.sin(theta)

    # print(theta, theta*180/np.pi)

    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = h / 2 * sina, h / 2 * cosa
    p1x, p1y = centerx - wx + hx, centery - wy - hy
    p2x, p2y = centerx + wx + hx, centery + wy - hy
    p3x, p3y = centerx + wx - hx, centery + wy + hy
    p4x, p4y = centerx - wx - hx, centery - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


@register_box_converter(QuadriBoxes, RotatedBoxes, force=True)
def qbox2rbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rotated boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    # TODO support tensor-based minAreaRect later
    original_shape = boxes.shape[:-1]
    points = boxes.cpu().numpy().reshape(-1, 4, 2)
    rboxes = []
    for pts in points:

        # calculate the center of the line
        x1, y1 = pts[0]
        x2, y2 = pts[1]

        theta = np.arctan2(y2 - y1, x2 - x1)

        w = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        h = np.sqrt((pts[2][0] - pts[1][0])**2 + (pts[2][1] - pts[1][1])**2)

        (x, y), (w1, h1), angle = cv2.minAreaRect(pts)
        # assert np.abs((angle / 180 * np.pi) - theta%(np.pi/2)) < 1e-4
        assert np.abs(w - w1) < 1e-2 or np.abs(w - h1) < 1e-2
        assert np.abs(h - w1) < 1e-2 or np.abs(h - h1) < 1e-2

        rboxes.append([x, y, w, h, theta])

    rboxes = boxes.new_tensor(rboxes)
    return rboxes.view(*original_shape, 5)
