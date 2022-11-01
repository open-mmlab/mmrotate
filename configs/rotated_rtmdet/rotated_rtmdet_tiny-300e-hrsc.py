_base_ = './rotated_rtmdet_s-300e-hrsc.py'

cocop = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth' # noqa
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=cocop),
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        # init_cfg=dict(
        #     type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
    ),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.CachedMosaic',
        img_scale=(800, 800),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        resize_type='mmdet.Resize',
        scale=(1600, 1600),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=(800, 800)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
