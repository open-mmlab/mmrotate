# Copyright (c) OpenMMLab. All rights reserved.
from .anchor_generator import (FakeRotatedAnchorGenerator,
                               PseudoRotatedAnchorGenerator)
from .utils import rotated_anchor_inside_flags

__all__ = [
    'FakeRotatedAnchorGenerator', 'PseudoRotatedAnchorGenerator',
    'rotated_anchor_inside_flags'
]
