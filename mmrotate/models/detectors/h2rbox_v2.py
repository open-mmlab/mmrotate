# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Tuple, Union

import torch
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from torch import Tensor
from torch.nn.functional import grid_sample
from torchvision import transforms

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes


@MODELS.register_module()
class H2RBoxV2Detector(SingleStageDetector):
    """Implementation of `H2RBox-v2 <https://arxiv.org/abs/2304.04403>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 crop_size: Tuple[int, int] = (768, 768),
                 padding: str = 'reflection',
                 view_range: Tuple[float, float] = (0.25, 0.75),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.crop_size = crop_size
        self.padding = padding
        self.view_range = view_range

    def rotate_crop(
            self,
            batch_inputs: Tensor,
            rot: float = 0.,
            size: Tuple[int, int] = (768, 768),
            batch_gt_instances: InstanceList = None,
            padding: str = 'reflection') -> Tuple[Tensor, InstanceList]:
        """

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            rot (float): Angle of view rotation. Defaults to 0.
            size (tuple[int]): Crop size from image center.
                Defaults to (768, 768).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            padding (str): Padding method of image black edge.
                Defaults to 'reflection'.

        Returns:
            Processed batch_inputs (Tensor) and batch_gt_instances
            (list[:obj:`InstanceData`])
        """
        device = batch_inputs.device
        n, c, h, w = batch_inputs.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2
        crop_w = (w - size_w) // 2
        if rot != 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = batch_inputs.new_tensor([[cosa, -sina], [sina, cosa]],
                                         dtype=torch.float)
            x_range = torch.linspace(-1, 1, w, device=device)
            y_range = torch.linspace(-1, 1, h, device=device)
            y, x = torch.meshgrid(y_range, x_range)
            grid = torch.stack([x, y], -1).expand([n, -1, -1, -1])
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
            # rotate
            batch_inputs = grid_sample(
                batch_inputs, grid, 'bilinear', padding, align_corners=True)
            if batch_gt_instances is not None:
                for i, gt_instances in enumerate(batch_gt_instances):
                    gt_bboxes = get_box_tensor(gt_instances.bboxes)
                    xy, wh, a = gt_bboxes[..., :2], gt_bboxes[
                        ..., 2:4], gt_bboxes[..., [4]]
                    ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + rot
                    rot_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                    batch_gt_instances[i].bboxes = RotatedBoxes(rot_gt_bboxes)
        batch_inputs = batch_inputs[..., crop_h:crop_h + size_h,
                                    crop_w:crop_w + size_w]
        if batch_gt_instances is None:
            return batch_inputs
        else:
            for i, gt_instances in enumerate(batch_gt_instances):
                gt_bboxes = get_box_tensor(gt_instances.bboxes)
                xy, wh, a = gt_bboxes[..., :2], gt_bboxes[...,
                                                          2:4], gt_bboxes[...,
                                                                          [4]]
                xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                batch_gt_instances[i].bboxes = RotatedBoxes(crop_gt_bboxes)

            return batch_inputs, batch_gt_instances

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances, _, _ = unpack_gt_instances(batch_data_samples)

        # Crop original images and gts
        batch_inputs, batch_gt_instances = self.rotate_crop(
            batch_inputs, 0, self.crop_size, batch_gt_instances, self.padding)
        offset = 1
        for gt_instances in batch_gt_instances:
            gt_instances.bid = torch.arange(
                0,
                len(gt_instances.bboxes),
                1,
                device=gt_instances.bboxes.device) + offset + 0.2
            offset += len(gt_instances.bboxes)

        # Generate rotated images and gts
        rot = math.pi * (
            torch.rand(1, device=batch_inputs.device) *
            (self.view_range[1] - self.view_range[0]) + self.view_range[0])
        batch_gt_rot = copy.deepcopy(batch_gt_instances)
        batch_inputs_rot, batch_gt_rot = self.rotate_crop(
            batch_inputs, rot, self.crop_size, batch_gt_rot, self.padding)
        offset = 1
        for gt_instances in batch_gt_rot:
            gt_instances.bid = torch.arange(
                0,
                len(gt_instances.bboxes),
                1,
                device=gt_instances.bboxes.device) + offset + 0.4
            offset += len(gt_instances.bboxes)

        # Generate flipped images and gts
        batch_inputs_flp = transforms.functional.vflip(batch_inputs)
        batch_gt_flp = copy.deepcopy(batch_gt_instances)
        offset = 1
        for gt_instances in batch_gt_flp:
            gt_instances.bboxes.flip_(batch_inputs.shape[2:4], 'vertical')
            gt_instances.bid = torch.arange(
                0,
                len(gt_instances.bboxes),
                1,
                device=gt_instances.bboxes.device) + offset + 0.6
            offset += len(gt_instances.bboxes)

        # Concat original/rotated/flipped images and gts
        batch_inputs_all = torch.cat(
            (batch_inputs, batch_inputs_rot, batch_inputs_flp))
        batch_data_samples_all = []
        for gt_instances in batch_gt_instances + batch_gt_rot + batch_gt_flp:
            data_sample = DetDataSample()
            data_sample.gt_instances = gt_instances
            batch_data_samples_all.append(data_sample)

        feat = self.extract_feat(batch_inputs_all)
        losses = self.bbox_head.loss(feat, batch_data_samples_all)

        return losses
