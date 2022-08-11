# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes, register_bbox_mode_converter
from torch import Tensor

from .quadrilateral_bbox import QuadriBoxes
from .rotated_bbox import RotatedBoxes


@register_bbox_mode_converter(HorizontalBoxes, RotatedBoxes)
def hbbox2rbbox(bboxes: Tensor) -> Tensor:
    """Convert horizontal boxes to rotated boxes.

    Args:
        bboxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    wh = bboxes[..., 2:] - bboxes[..., :2]
    ctrs = (bboxes[..., 2:] + bboxes[..., :2]) / 2
    theta = bboxes.new_zeros((*bboxes.shape[:-1], 1))
    return torch.cat([ctrs, wh, theta], dim=-1)


@register_bbox_mode_converter(HorizontalBoxes, QuadriBoxes)
def hbbox2qbbox(bboxes: Tensor) -> Tensor:
    """Convert horizontal boxes to quadrilateral boxes.

    Args:
        bboxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    x1, y1, x2, y2 = torch.split(bboxes, 1, dim=-1)
    return torch.cat([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1)


@register_bbox_mode_converter(RotatedBoxes, HorizontalBoxes)
def rbbox2hbbox(bboxes: Tensor) -> Tensor:
    """Convert rotated boxes to horizontal boxes.

    Args:
        bboxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    ctrs, w, h, theta = torch.split(bboxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * cos_value) + torch.abs(h / 2 * sin_value)
    y_bias = torch.abs(w / 2 * sin_value) + torch.abs(h / 2 * cos_value)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([ctrs - bias, ctrs + bias], dim=-1)


@register_bbox_mode_converter(RotatedBoxes, QuadriBoxes)
def rbbox2qbbox(bboxes: Tensor) -> Tensor:
    """Convert rotated boxes to quadrilateral boxes.

    Args:
        bboxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    ctr, w, h, theta = torch.split(bboxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
    vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.cat([pt1, pt2, pt3, pt4], dim=-1)


@register_bbox_mode_converter(QuadriBoxes, HorizontalBoxes)
def qbbox2hbbox(bboxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to horizontal boxes.

    Args:
        bboxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    bboxes = bboxes.view(*bboxes.shape[:-1], 4, 2)
    x1y1, _ = bboxes.min(dim=-2)
    x2y2, _ = bboxes.max(dim=-2)
    return torch.cat([x1y1, x2y2], dim=-1)


@register_bbox_mode_converter(QuadriBoxes, RotatedBoxes)
def qbbox2rbbox(bboxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rotated boxes.

    Args:
        bboxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    original_shape = bboxes.shape[:-1]
    points = bboxes.cpu().numpy().reshape(-1, 4, 2)
    rbboxes = []
    for pts in points:
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rbboxes.append([x, y, w, h, angle / 180 * np.pi])
    rbboxes = bboxes.new_tensor(rbboxes)
    return rbboxes.view(*original_shape, 5)
