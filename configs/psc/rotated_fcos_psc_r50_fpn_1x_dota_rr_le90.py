_base_ = '../rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90.py'

angle_version = 'le90'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=[(768, 768), (1024, 1024)]),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version=angle_version),
    dict(type='PolyRandomRotate',
         rotate_ratio=0.5,
         angles_range=180,
         auto_bound=False,
         rect_classes=[9, 11],
         version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

model = dict(bbox_head=dict(type='PSCRFCOSHead',
                            center_sampling=True,
                            center_sample_radius=1.5,
                            norm_on_bbox=True,
                            centerness_on_reg=True,
                            separate_angle=True,
                            scale_angle=False,
                            angle_coder=dict(type='PSCCoder',
                                             angle_version=angle_version,
                                             dual_freq=False,
                                             num_step=3),
                            loss_cls=dict(type='FocalLoss',
                                          use_sigmoid=True,
                                          gamma=2.0,
                                          alpha=0.25,
                                          loss_weight=1.0),
                            loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                            loss_centerness=dict(type='CrossEntropyLoss',
                                                 use_sigmoid=True,
                                                 loss_weight=1.0),
                            loss_angle=dict(type='MSELoss',
                                            loss_weight=0.2)), )
