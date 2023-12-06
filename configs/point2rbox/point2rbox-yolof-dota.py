_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='Point2RBoxYOLOF',
    crop_size=(1024, 1024),
    prob_rot=0.95 * 0.7,
    prob_flp=0.05 * 0.7,
    sca_fact=0.4,
    sca_range=(0.5, 1.5),
    basic_pattern='data/basic_patterns/dota',
    dense_cls=[4, 5, 6, 9],
    use_setrc=False,
    use_setsk=True,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        strides=(1, 2, 2, 1),  # DC5
        dilations=(1, 1, 1, 2),
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='mmdet.DilatedEncoder',
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4,
        block_dilations=[2, 4, 6, 8]),
    bbox_head=dict(
        type='Point2RBoxYOLOFHead',
        num_classes=15,
        in_channels=512,
        reg_decoded_bbox=True,
        num_cls_convs=4,
        num_reg_convs=8,
        use_objectness=False,
        agnostic_cls=[1, 9, 11],
        square_cls=[0],
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            ratios=[1.0],
            scales=[4, 4, 4, 4, 4],
            strides=[16]),
        bbox_coder=dict(
            type='mmdet.DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=16),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.6),
        loss_scale_ss=dict(type='mmdet.GIoULoss', loss_weight=0.04)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='Point2RBoxAssigner',
            pos_ignore_thr=0.15,
            neg_ignore_thr=0.7,
            match_times=2),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RBox2Point'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.RandomShift', prob=0.5, max_shift_px=16),
    dict(type='mmdet.PackDetInputs')
]

train_cfg = dict(type='EpochBasedTrainLoop', val_interval=12)

train_dataloader = dict(
    batch_size=4, num_workers=4, dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(batch_size=4, num_workers=4)

val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=[0.25, 0.5])

# default_hooks = dict(logger=dict(type='LoggerHook', interval=30))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
