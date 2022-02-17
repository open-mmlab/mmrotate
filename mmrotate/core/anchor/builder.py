# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import build_from_cfg
from mmdet.core.anchor.builder import ANCHOR_GENERATORS

ROTATED_ANCHOR_GENERATORS = ANCHOR_GENERATORS


def build_prior_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ROTATED_ANCHOR_GENERATORS, default_args)
