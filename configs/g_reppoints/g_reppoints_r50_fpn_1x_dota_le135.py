_base_ = ['../rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py']

angle_version = 'le135'

model = dict(
    bbox_head=dict(
        version=angle_version,
        type='KLDRepPointsHead',
        loss_bbox_init=dict(type='KLDRepPointsLoss'),
        loss_bbox_refine=dict(type='KLDRepPointsLoss')),
    train_cfg=dict(
        refine=dict(
            assigner=dict(_delete_=True, type='ATSSKldAssigner', topk=9))))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
