_base_ = 'rotated_atss_obb_r50_fpn_1x_dota_le90.py'

angle_version = 'oc'

model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=angle_version),
        bbox_coder=dict(
            angle_version=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True)),
    train_cfg=dict(
        assigner=dict(iou_calculator=dict(type='FakeRBboxOverlaps2D'))))

train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
