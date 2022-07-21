from sonic_ai.pipelines.init_pipeline import LoadCategoryList

_base_ = [
    '../base/dotav1.py', '../base/schedule_1x.py',
    '../base/default_runtime.py',
    '../base/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
]
data_root = '/data/xys/data/byd_ftj/'

label_path = '/data2/5-标注数据/0-分条机-CYS.220215-赢合-比亚迪分条机/label.ini'

category_list = LoadCategoryList()(results={
    'label_path': label_path
})['category_list']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
angle_version = 'le90'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

optimizer = dict(lr=0.005)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        category_list=category_list,
        ann_file=data_root + 'train/labelTxt/',
        img_prefix=data_root + 'train/images/',
        version=angle_version),
    val=dict(
        category_list=category_list,
        ann_file=data_root + 'test/labelTxt/',
        img_prefix=data_root + 'test/images/',
        version=angle_version),
    test=dict(
        category_list=category_list,
        ann_file=data_root + 'test/labelTxt/',
        img_prefix=data_root + 'test/images/',
        version=angle_version))

model = dict(roi_head=dict(bbox_head=dict(num_classes=len(category_list))))

LoadCategoryList = None
