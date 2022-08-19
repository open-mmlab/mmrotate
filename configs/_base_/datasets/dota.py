# dataset settings
dataset_type = 'DOTADataset'
data_root = 'data/split_ss_dota/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
<<<<<<< HEAD
<<<<<<< HEAD
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
=======
    dict(type='mmdet.Resize', scale=(1024, 2014), keep_ratio=True),
>>>>>>> 61dcdf7 (init)
=======
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
>>>>>>> bc74907 (fix size typo)
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
<<<<<<< HEAD
<<<<<<< HEAD
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
=======
    dict(type='mmdet.Resize', scale=(1024, 2014), keep_ratio=True),
>>>>>>> 61dcdf7 (init)
=======
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
>>>>>>> bc74907 (fix size typo)
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
<<<<<<< HEAD
<<<<<<< HEAD
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
=======
    dict(type='mmdet.Resize', scale=(1024, 2014), keep_ratio=True),
>>>>>>> 61dcdf7 (init)
=======
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
>>>>>>> bc74907 (fix size typo)
    # avoid bboxes being resized
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=(1024, 1024),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

<<<<<<< HEAD
<<<<<<< HEAD
# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
=======
# inference on test dataset and
# format the output results for submission.
>>>>>>> 61dcdf7 (init)
=======
# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
>>>>>>> 0289589 (update configs & RBboxOverlaps2D & FakeRBboxOverlaps2D)
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
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
#     outfile_prefix='./work_dirs/dota/Task1')
