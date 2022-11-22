_base_ = \
    ['../rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_rr-6x_hrsc.py']

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=None),
        type='AngleBranchRetinaHead',
        use_normalized_angle_feat=True,
        angle_coder=dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=False,
            num_step=3,
            thr_mod=0.0),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.7),
        loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.6)))
