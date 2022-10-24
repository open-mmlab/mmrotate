_base_ = 'rotated-fcos-le90_r50_fpn_1x_dota.py'

model = dict(
    bbox_head=dict(
        use_hbbox_loss=True,
        scale_angle=True,
        angle_coder=dict(type='PseudoAngleCoder'),
        loss_angle=dict(_delete_=True, type='mmdet.L1Loss', loss_weight=0.2),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
    ))
