# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, BBOX_CODERS, BBOX_SAMPLERS

ROTATED_BBOX_ASSIGNERS = BBOX_ASSIGNERS
ROTATED_BBOX_SAMPLERS = BBOX_SAMPLERS
ROTATED_BBOX_CODERS = BBOX_CODERS


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    return build_from_cfg(cfg, ROTATED_BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, ROTATED_BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, ROTATED_BBOX_CODERS, default_args)
