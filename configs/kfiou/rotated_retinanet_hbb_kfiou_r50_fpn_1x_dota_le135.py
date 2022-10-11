_base_ = '../rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_le135.py'

model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        loss_bbox_type='kfiou',
        loss_bbox=dict(type='KFLoss', loss_weight=5.0)))
