_base_ = ['../r3det/r3det_r50_fpn_1x_dota_oc.py']

angle_version = 'oc'
model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1.0,
            loss_weight=1.0)),
    refine_heads=[
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=15,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            reg_decoded_bbox=True,
            loss_bbox=dict(
                type='GDLoss_v1',
                loss_type='kld',
                fun='log1p',
                tau=1.0,
                loss_weight=1.0))
    ])
