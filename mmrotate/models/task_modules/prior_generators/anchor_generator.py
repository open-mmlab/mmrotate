# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmdet.models.task_modules import AnchorGenerator
from mmdet.structures.bbox import HorizontalBoxes
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes


@TASK_UTILS.register_module()
class FakeRotatedAnchorGenerator(AnchorGenerator):
    """Fake rotate anchor generator for 2D anchor-based detectors. Horizontal
    bounding box represented by (x,y,w,h,theta).

    Note: In mmrotate-0.x, the angle of anchor is always 0. If you want to
    load models in 0.x directly, please set the `angle_version` to 'None'.

    Args:
        angle_version (str, optional): Angle definition of rotated bbox.
            Can only be 'None', 'oc', 'le90', or 'le135'. 'None' means the
            angle of anchor is always 0. Defaults to None.
    """

    def __init__(self, angle_version: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.angle_version = angle_version

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: str = 'cuda') -> Tensor:
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (str): The device the tensor will be put on.
                Defaults to ``cuda``.
        Returns:
            Tensor: Anchors in the overall feature maps.
        """
        anchors = super().single_level_grid_priors(
            featmap_size, level_idx, dtype=dtype, device=device)
        anchors = HorizontalBoxes(anchors, clone=False)
        anchors = anchors.convert_to('rbox')
        if self.angle_version:
            anchors = anchors.regularize_boxes(self.angle_version)
            anchors = RotatedBoxes(anchors, clone=False)
        return anchors


@TASK_UTILS.register_module()
class PseudoRotatedAnchorGenerator(AnchorGenerator):
    """Non-Standard pseudo anchor generator that is used to generate valid
    flags only!"""

    def __init__(self, strides: List[int]) -> None:
        self.strides = [_pair(stride) for stride in strides]

    @property
    def num_base_priors(self) -> None:
        """list[int]: total number of base priors in a feature grid"""
        return [1 for _ in self.strides]
