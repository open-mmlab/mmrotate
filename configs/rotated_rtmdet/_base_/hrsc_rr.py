# dataset settings
dataset_type = 'HRSCDataset'
data_root = 'data/hrsc/'
backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    dict(type='mmdet.Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='ImageSets/trainval.txt',
            data_prefix=dict(sub_data_root='FullDataSet/'),
            filter_cfg=dict(filter_empty_gt=True),
            pipeline=train_pipeline,
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/test.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='DOTAMetric',
        eval_mode='11points',
        prefix='dota_ap07',
        metric='mAP'),
    dict(
        type='DOTAMetric', eval_mode='area', prefix='dota_ap12', metric='mAP'),
]
test_evaluator = val_evaluator
