# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.utils import to_2tuple
from mmdet.models.task_modules import AnchorGenerator
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from ..bbox import hbb2obb


@TASK_UTILS.register_module()
class FakeRotatedAnchorGenerator(AnchorGenerator):
    """Fake rotate anchor generator for 2D anchor-based detectors. Horizontal
    bounding box represented by (x,y,w,h,theta).

    Args:
        angle_version (str): Angle definition of rotated bbox.
            Defaults to 'oc'.
    """

    def __init__(self, angle_version: str = 'oc', **kwargs) -> None:
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

        anchors = hbb2obb(anchors, self.angle_version)

        return anchors


@TASK_UTILS.register_module()
class PseudoRotatedAnchorGenerator(AnchorGenerator):
    """Non-Standard pseudo anchor generator that is used to generate valid
    flags only!"""

    def __init__(self, strides: List[int]) -> None:
        self.strides = [to_2tuple(stride) for stride in strides]

    @property
    def num_base_anchors(self) -> None:
        """list[int]: total number of base anchors in a feature grid"""
        return [1 for _ in self.strides]

    def single_level_grid_anchors(self,
                                  base_anchors: Tensor,
                                  featmap_size: Tuple[int],
                                  stride: Tuple[int],
                                  device: str = 'cuda') -> None:
        """Calling its grid_anchors() method will raise NotImplementedError!"""
        raise NotImplementedError

    def __repr__(self) -> str:
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides})'
        return repr_str
