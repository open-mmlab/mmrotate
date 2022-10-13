# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from torch import Tensor

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class CSLCoder(BaseBBoxCoder):
    """Circular Smooth Label Coder.

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        angle_version (str): Angle definition.
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, angle_version, omega=1, window='gaussian', radius=6):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        assert window in ['gaussian', 'triangle', 'rect', 'pulse']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        self.omega = omega
        self.window = window
        self.radius = radius
        self.encode_size = int(self.angle_range // omega)

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level
                Has shape (num_anchors * H * W, 1)

        Returns:
            Tensor: The csl encoding of angle offset for each scale
            level. Has shape (num_anchors * H * W, encoded_size)
        """

        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.encode_size)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.encode_size
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = torch.exp(-torch.pow(base_radius_range, 2) /
                                     (2 * self.radius**2))

        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)

        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset for each
                scale level. Has shape (num_anchors * H * W, encoded_size) or
                (B, num_anchors * H * W, encoded_size)
            keepdim (bool): Whether the output tensor has dim retained or not.


        Returns:
            Tensor: Angle offset for each scale level. When keepdim is true,
            return (num_anchors * H * W, 1) or (B, num_anchors * H * W, 1),
            otherwise (num_anchors * H * W,) or (B, num_anchors * H * W)
        """
        if angle_preds.shape[0] == 0:
            shape = list(angle_preds.size())
            if keepdim:
                shape[-1] = 1
            else:
                shape = shape[:-1]
            return angle_preds.new_zeros(shape)
        angle_cls_inds = torch.argmax(angle_preds, dim=-1, keepdim=keepdim)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)


@TASK_UTILS.register_module()
class PseudoAngleCoder(BaseBBoxCoder):
    """Pseudo Angle Coder."""

    encode_size = 1

    def encode(self, angle_targets: Tensor) -> Tensor:
        return angle_targets

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        if keepdim:
            return angle_preds
        else:
            return angle_preds.squeeze(-1)


@TASK_UTILS.register_module()
class DistributionAngleCoder(BaseBBoxCoder):

    def __init__(self, angle_version='le90', reg_max=16):
        super().__init__()
        self.angle_range = 0.5 * np.pi if angle_version == 'oc' else np.pi
        self.angle_offset_dict = {
            'oc': 0,
            'le90': 0.5 * np.pi,
            'le135': 0.25 * np.pi
        }
        self.angle_offset = self.angle_offset_dict[angle_version]
        self.reg_max = reg_max
        self.encode_size = reg_max + 1
        self.project = torch.linspace(0, self.reg_max, self.reg_max + 1)

    def encode(self, angle):
        # Norm to (0~1)*reg_max
        dfl_target = self.reg_max * (self.angle_offset +
                                     angle) / self.angle_range
        return dfl_target.flatten()

    def decode(self, angle, keepdim=True):
        angle = F.softmax(angle.reshape(-1, self.reg_max + 1), dim=-1)
        angle = F.linear(angle, self.project.type_as(angle)).reshape(-1, 1)
        return self.angle_range * angle / self.reg_max - self.angle_offset
