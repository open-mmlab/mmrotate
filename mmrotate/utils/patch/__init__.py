# Copyright (c) OpenMMLab. All rights reserved.
from .merge_results import merge_results_by_nms
from .split import get_multiscale_patch, slide_window

__all__ = ['merge_results_by_nms', 'get_multiscale_patch', 'slide_window']
