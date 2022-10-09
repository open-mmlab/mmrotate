_base_ = \
    ['../rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py']

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        type='AngleBranchRetinaHead',
        angle_coder=dict(
            type='CSLCoder',
            angle_version=angle_version,
            omega=4,
            window='gaussian',
            radius=3),
        loss_angle=dict(
            type='mmdet.SmoothFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.8)))
