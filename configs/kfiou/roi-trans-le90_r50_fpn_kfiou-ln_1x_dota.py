_base_ = '../roi_trans/roi-trans-le90_r50_fpn_1x_dota.py'

angle_version = 'le90'
model = dict(
    roi_head=dict(bbox_head=[
        dict(
            type='RotatedShared2FCBBoxHead',
            predict_box_type='rbox',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTHBBoxCoder',
                angle_version=angle_version,
                norm_factor=2,
                edge_swap=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
                use_box_type=True),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox_type='kfiou',
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0)),
        dict(
            type='RotatedShared2FCBBoxHead',
            predict_box_type='rbox',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1, 0.05]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox_type='kfiou',
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0))
    ]))
