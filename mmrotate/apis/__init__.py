# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_detector_by_patches
from .patch import get_multiscale_patch, merge_results_by_nms, slide_window

__all__ = [
    'inference_detector_by_patches', 'get_multiscale_patch', 'slide_window',
    'merge_results_by_nms'
]
