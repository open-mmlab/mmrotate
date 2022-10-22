# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import mmcv
from mmcv.transforms import BaseTransform

from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPatchFromNDArray(BaseTransform):
    """Load a patch from the huge image w.r.t ``results['patch']``.

    Requaired Keys:

    - img
    - patch

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        pad_val (float or Sequence[float]): Values to be filled in padding
            areas. Defaults to 0.
    """

    def __init__(self,
                 pad_val: Union[float, Sequence[float]] = 0,
                 **kwargs) -> None:
        self.pad_val = pad_val

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with image array in ``results['img']``
                and patch position in ``results['patch']``.

        Returns:
            dict: The dict contains loaded patch and meta information.
        """
        image = results['img']
        img_h, img_w = image.shape[:2]

        patch_xmin, patch_ymin, patch_xmax, patch_ymax = results['patch']
        assert (patch_xmin < img_w) and (patch_xmax >= 0) and \
            (patch_ymin < img_h) and (patch_ymax >= 0)
        x1 = max(patch_xmin, 0)
        y1 = max(patch_ymin, 0)
        x2 = min(patch_xmax, img_w)
        y2 = min(patch_ymax, img_h)
        padding = (x1 - patch_xmin, y1 - patch_ymin, patch_xmax - x2,
                   patch_ymax - y2)

        patch = image[y1:y2, x1:x2]
        if any(padding):
            patch = mmcv.impad(patch, padding=padding, pad_val=self.pad_val)

        results['img_path'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape[:2]
        results['ori_shape'] = patch.shape[:2]
        return results
