# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models.builder import BACKBONES

ROTATED_BACKBONES = BACKBONES
ROTATED_LOSSES = BACKBONES
ROTATED_DETECTORS = BACKBONES
ROTATED_ROI_EXTRACTORS = BACKBONES
ROTATED_HEADS = BACKBONES
ROTATED_NECKS = BACKBONES
ROTATED_SHARED_HEADS = BACKBONES


def build_backbone(cfg):
    """Build backbone."""
    return ROTATED_BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return ROTATED_NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROTATED_ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return ROTATED_SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return ROTATED_HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return ROTATED_LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return ROTATED_DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
