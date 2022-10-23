_base_ = \
    ['../rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota.py']

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
            type='SmoothFocalLoss', gamma=2.0, alpha=0.25, loss_weight=0.8)))
