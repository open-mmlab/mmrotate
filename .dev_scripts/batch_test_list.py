# Copyright (c) OpenMMLab. All rights reserved.

# Note: mAP is on DOTA trainval set here.

# yapf: disable

cfa = dict(
    config='configs/cfa/cfa-qbox_r50_fpn_1x_dota.py',
    checkpoint='atss_r50_fpn_1x_coco_20200209-985f7bd0.pth',
    url='https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_1x_dota_le135/cfa_r50_fpn_1x_dota_le135-aed1cbc6.pth', # noqa
    metric=dict(mAP=69.63),
)
# yapf: enable
