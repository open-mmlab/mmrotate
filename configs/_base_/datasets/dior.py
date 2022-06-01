# dataset settings
dataset_type = 'DIORDataset'
data_root = '/cluster/home/it_stu7/main/datasets/DIOR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[data_root + 'Main/train.txt', data_root + 'Main/val.txt'],
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_subdir=data_root + 'JPEGImages-trainval/',
        img_prefix=data_root + 'JPEGImages-trainval/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Main/test.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_subdir=data_root + 'JPEGImages-test/',
        img_prefix=data_root + 'JPEGImages-test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Main/test.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_subdir=data_root + 'JPEGImages-test/',
        img_prefix=data_root + 'JPEGImages-test/',
        pipeline=test_pipeline))
