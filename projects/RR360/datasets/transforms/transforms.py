# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

# import cv2
import mmcv

from mmrotate.datasets.transforms import Rotate
from mmrotate.registry import TRANSFORMS

# import numpy as np
# from mmcv.transforms import BaseTransform
# from mmcv.transforms.utils import cache_randomness
# from mmdet.structures.bbox import BaseBoxes, get_box_type
# from mmdet.structures.mask import PolygonMasks
# from mmengine.utils import is_list_of


@TRANSFORMS.register_module()
class RotateAutoBound(Rotate):
    """Rotate the images, bboxes, masks and segmentation map by a certain
    angle. Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        rotate_angle (int): An angle to rotate the image.
        img_border_value (int or float or tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 0.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 rotate_angle: int,
                 img_border_value: Union[int, float, tuple] = 0,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        if isinstance(img_border_value, (float, int)):
            img_border_value = tuple([float(img_border_value)] * 3)
        elif isinstance(img_border_value, tuple):
            assert len(img_border_value) == 3, \
                f'img_border_value as tuple must have 3 elements, ' \
                f'got {len(img_border_value)}.'
            img_border_value = tuple([float(val) for val in img_border_value])
        else:
            raise ValueError(
                'img_border_value must be float or tuple with 3 elements.')
        self.rotate_angle = rotate_angle
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label
        self.interpolation = interpolation

    def _transform_img(self, results: dict) -> None:
        """Rotate the image."""
        results['img'] = mmcv.imrotate(
            results['img'],
            self.rotate_angle,
            auto_bound=True,
            border_value=self.img_border_value,
            interpolation=self.interpolation)
        results['img_shape_record'] = [[
            results['img_shape'][0], results['img_shape'][1]
        ], [results['img'].shape[0], results['img'].shape[1]]]
        results['img_shape'] = [
            results['img'].shape[0], results['img'].shape[1]
        ]

    def _transform_bboxes(self, results: dict) -> None:
        """Rotate the bboxes."""
        if len(results['gt_bboxes']) == 0:
            return
        img_shape = results['img_shape']
        center = (img_shape[1] * 0.5, img_shape[0] * 0.5)
        results['gt_bboxes'].rotate_auto_bound_(center, self.rotate_angle,
                                                results['img_shape_record'])
        results['gt_bboxes'].clip_(img_shape)
