_base_ = '../rotated_retinanet/rotated-retinanet-rbox-le135_r50_fpn_1x_dota.py'

model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        loss_bbox_type='kfiou',
        loss_bbox=dict(type='KFLoss', loss_weight=5.0)),
    train_cfg=dict(
        assigner=dict(iou_calculator=dict(type='FakeRBboxOverlaps2D'))))
