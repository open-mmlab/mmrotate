_base_ = '../rotated_fcos/rotated-fcos-le90_r50_fpn_1x_dota.py'

angle_version = {{_base_.angle_version}}

# model settings
model = dict(
    bbox_head=dict(
        use_hbbox_loss=True,
        scale_angle=False,
        angle_coder=dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=True,
            num_step=3),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_angle=dict(_delete_=True, type='mmdet.L1Loss', loss_weight=0.1),
    ))
