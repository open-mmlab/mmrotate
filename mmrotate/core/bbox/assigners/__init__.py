# Copyright (c) OpenMMLab. All rights reserved.
from .atss_kld_assigner import ATSSKldAssigner
from .convex_assigner import ConvexAssigner
from .max_convex_iou_assigner import MaxConvexIoUAssigner
from .sas_assigner import SASAssigner

__all__ = [
    'ConvexAssigner', 'MaxConvexIoUAssigner', 'SASAssigner', 'ATSSKldAssigner'
]
