# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_overlaps import bbox_overlaps, fake_rbbox_overlaps
from .mean_ap import eval_rbbox_map

__all__ = ['bbox_overlaps', 'fake_rbbox_overlaps', 'eval_rbbox_map']
