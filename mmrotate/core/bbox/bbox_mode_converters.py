# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmdet.structures.bbox import register_bbox_mode_converter
from torch import Tensor  # noqa

from .quadrilateral_bbox import QuadriBoxes  # noqa
from .rotated_bbox import RotatedBoxes  # noqa


@register_bbox_mode_converter('hbbox', 'rbbox')
def hbbox2rbbox(bboxes):
    theta = bboxes.new_zeros((*bboxes.shape[:-1], 1))
    return torch.cat([bboxes, theta], dim=-1)


@register_bbox_mode_converter('hbbox', 'qbbox')
def hbbox2qbbox(bboxes):
    x1, y1, x2, y2 = torch.split(bboxes, 1, dim=-1)
    return torch.cat([x1, y1, x2, y1, x1, y2, x2, y2], dim=-1)


@register_bbox_mode_converter('rbbox', 'hbbox')
def rbbox2hbbox(bboxes):
    ctrs, w, h, theta = torch.split(bboxes, (2, 1, 1, 1), dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([ctrs - bias, ctrs + bias], dim=-1)


@register_bbox_mode_converter('rbbox', 'qbbox')
def rbbox2qbbox(bboxes):
    ctr, w, h, theta = torch.split(bboxes, (2, 1, 1, 1), dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    vec1 = torch.cat([w / 2 * Cos, w / 2 * Sin], dim=-1)
    vec2 = torch.cat([-h / 2 * Sin, h / 2 * Cos], dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.cat([pt1, pt2, pt3, pt4], dim=-1)


@register_bbox_mode_converter('qbbox', 'hbbox')
def qbbox2hbbox(bboxes):
    bboxes = bboxes.view(*bboxes.shape[:-1], 4, 2)
    x1y1 = bboxes.min(dim=-2)
    x2y2 = bboxes.max(dim=-2)
    return torch.cat([x1y1, x2y2], dim=-1)


@register_bbox_mode_converter('qbbox', 'rbbox')
def qbbox2rbbox(bboxes):
    original_shape = bboxes.shape[:-1]
    points = bboxes.cpu().numpy().reshape(-1, 4, 2)
    rbboxes = []
    for pts in points:
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rbboxes.append([x, y, w, h, angle / 180 * np.pi])
    rbboxes = bboxes.new_tensor(rbboxes)
    return rbboxes.view(*original_shape, 5)
