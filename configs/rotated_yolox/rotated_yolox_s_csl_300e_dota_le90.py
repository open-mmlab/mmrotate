_base_ = './rotated_yolox_s_300e_dota_le90.py'

# model settings
model = dict(
    bbox_head=dict(
        separate_angle=True,
        with_angle_l1=False,
        loss_bbox=dict(
            _delete_=True,
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        angle_coder=dict(
            type='CSLCoder',
            angle_version='le90',
            omega=1,
            window='gaussian',
            radius=2),
        loss_angle=dict(
            type='SmoothFocalLoss',
            gamma=2.0,
            alpha=0.25,
            reduction='sum',
            loss_weight=1.0)))
