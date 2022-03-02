# Copyright (c) OpenMMLab. All rights reserved.
from .merge_results import merge_results
from .split import get_multiscale_patch, slide_window

__all__ = ['merge_results', 'get_multiscale_patch', 'slide_window']
