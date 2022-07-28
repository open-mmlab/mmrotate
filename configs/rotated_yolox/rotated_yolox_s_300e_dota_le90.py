_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

angle_version = 'le90'
img_scale = (1024, 1024)  # height, width

# model settings
model = dict(
    type='RotatedYOLOX',
    input_size=img_scale,
    random_size_range=(25, 35),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='RotatedYOLOXHead',
        num_classes=15,
        in_channels=128,
        feat_channels=128,
        seprate_angle=False,
        loss_bbox=dict(
            type='RotatedIoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type='RSimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(
        score_thr=0.01, nms=dict(type='nms_rotated', iou_threshold=0.10)))

# dataset settings
dataset_type = 'DOTADataset'
data_root = '/datasets/dota_mmrotate_ss/'

train_pipeline = [
    dict(type='RMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='PolyRandomAffine',
        version=angle_version,
        scaling_ratio_range=(0.1, 2),
        bbox_clip_border=False,
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='PolyMixUp',
        version=angle_version,
        bbox_clip_border=False,
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmrotate/models/detectors/rotated_yolox.py.'
    dict(type='RResize', img_scale=img_scale),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='FilterRotatedAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        version=angle_version,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        # Unlike the original implementation,
        # filter_empty_gt=False will lead to instability
        # at the end of the training and performance degradation
        filter_empty_gt=True,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        version=angle_version,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        version=angle_version,
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01 / 8,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 10

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        skip_type_keys=('RMosaic', 'PolyRandomAffine', 'PolyMixUp'),
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='mAP')
log_config = dict(interval=50)
