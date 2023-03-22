# dataset settings
dataset_type = 'DOTADataset'
data_root = 'data/TRR360D/'
backend_args = None
METAINFO = dict(classes=('table', ))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='RandomRotate',
        prob=1.0,
        rotate_type='RotateAutoBound',
        angle_range=180,
        rect_obj_labels=[9, 11]),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.0,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            metainfo=METAINFO,
            data_root=data_root,
            ann_file='ann_train_obbox/',
            data_prefix=dict(img_path='img_train_obbox/'),
            img_shape=(1024, 1024),
            filter_cfg=dict(filter_empty_gt=True),
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_root=data_root,
        # ann_file='ann_test_obbox/',
        ann_file='ann_test_obbox/',
        data_prefix=dict(img_path='img_test_obbox/'),
        # data_prefix=dict(img_path='img_test_obbox/'),
        img_shape=(1024, 1024),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='DOTAR360Metric', metric='mAP', iou_thrs=[0.5], angle_thr=90),
    dict(type='DOTAR360Metric', metric='mAP', iou_thrs=[0.5], angle_thr=360),
    dict(type='DOTAMetric', metric='mAP', iou_thrs=[0.5])
]
test_evaluator = [
    dict(
        type='DOTAR360Metric',
        metric='mAP',
        iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        angle_thr=90),
    dict(
        type='DOTAMetric',
        metric='mAP',
        iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
]

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=8,
#     num_workers=8,
#     persistent_workers=False,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='test/images/'),
#         img_shape=(1024, 1024),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix='./work_dirs/rtmdet_r/Task1')
