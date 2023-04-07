_base_ = './rotated_rtmdet_l-300e-aug-hrsc.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(
        in_channels=96,
        feat_channels=96,
        exp_on_reg=False,
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
    ))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.CachedMosaic',
        img_scale=(800, 800),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='mmdet.RandomResize',
        resize_type='mmdet.Resize',
        scale=(1600, 1600),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    dict(type='mmdet.RandomCrop', crop_size=(800, 800)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.CachedMixUp',
        img_scale=(800, 800),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomResize',
        resize_type='mmdet.Resize',
        scale=(800, 800),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    dict(type='mmdet.RandomCrop', crop_size=(800, 800)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=90,
        switch_pipeline=train_pipeline_stage2)
]
