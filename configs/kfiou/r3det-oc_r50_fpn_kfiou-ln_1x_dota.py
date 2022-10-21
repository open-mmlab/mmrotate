_base_ = ['../r3det/r3det-oc_r50_fpn_1x_dota.py']

angle_version = 'oc'
model = dict(
    bbox_head_init=dict(
        loss_bbox_type='kfiou',
        loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0)),
    bbox_head_refine=[
        dict(
            type='R3RefineHead',
            num_classes=15,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            frm_cfg=dict(
                type='FRM', feat_channels=256, strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox_type='kfiou',
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0))
    ])
