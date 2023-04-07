_base_ = './rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py'

model = dict(
    train_cfg=dict(
        assigner=dict(iou_calculator=dict(type='FakeRBboxOverlaps2D'))))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
