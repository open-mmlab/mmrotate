
# # _base = './rotated-retinanet-rbox-h180_r50_fpn_6x_rl1_dota_pt.py'
# _base = './rotated-retinanet-rbox-h180_r50_fpn_6x_rl1_dota.py'

# # model = dict(
# #     bbox_head=dict(
# #         loss_bbox=dict(
# #             loss_weight=10.0
# #         )
# #     )
# # )


_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'h180'

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=3, save_best=['dota/AP50'], rule='greater', max_keep_ckpts=1),
)

# custom_imports = dict(imports=['mmcls.models', 'projects.headet.visualization', 'projects.headet.structures'], allow_failed_imports=False)

# default_hooks = dict(
#     logger=dict(interval=1),
#     # visualization=dict(draw=True, interval=10),
# )

model = dict(
    type='mmdet.RetinaNet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='mmdet.RetinaHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version=angle_version,
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RL1Loss', loss_weight=5.0)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        sampler=dict(
            type='mmdet.PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))


load_from = 'https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth'