# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch
from mmdet.models import DetDataPreprocessor

from mmrotate.registry import MODELS


@MODELS.register_module()
class RotDataPreprocessor(DetDataPreprocessor):
    """Data pre-processor for rotated object detection tasks.

    Comparing with the :class:`mmdet.DetDataPreprocessor`,

    1. It supports convert bbox from 'qbb' to 'rbb'.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.
    - Convert bbox from 'qbb' to 'rbb'.


    Args:
        angle_version (str, Optional): Angle definition of 'rbb'. Can
        only be 'oc', 'le90', or 'le135'. Defaults to None, which means
        don't convert bbox from 'qbb' to 'rbb'.

    """

    def __init__(self,
                 angle_version: Optional(str) = None,
                 **kwargs):
        super().__init__(**kwargs)
        if angle_version is not None:
            assert angle_version in ['oc', 'le90', 'le135'], \
            'Unrecognized version, only "oc", "le90", and "le135" are supported'
        self.angle_version = angle_version

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """Perform bbox conversion based on ``DetDataPreprocessor``.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        batch_inputs, batch_data_samples = super().forward(
            data=data, training=training)

        if self.angle_version is None:
            return batch_inputs, batch_data_samples

        for data_samples in batch_data_samples:
            data_samples.gt_instances.bboxes.convert_to('rbox')
            data_samples.gt_instances.bboxes.regularize_boxes(self.angle_version)
            
        return batch_inputs, batch_data_samples
