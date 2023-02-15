_base_ = 'rotated-fcos-le90_r50_fpn_1x_dota.py'

angle_version = {{_base_.angle_version}}

# model settings
model = dict(
    bbox_head=dict(
        use_hbbox_loss=True,
        scale_angle=False,
        angle_coder=dict(
            type='CSLCoder',
            angle_version=angle_version,
            omega=1,
            window='gaussian',
            radius=1),
        loss_angle=dict(
            _delete_=True,
            type='SmoothFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
    ))
