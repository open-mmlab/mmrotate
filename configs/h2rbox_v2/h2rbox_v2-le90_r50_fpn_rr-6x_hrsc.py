_base_ = ['h2rbox_v2-le90_r50_fpn-6x_hrsc.py']

# load hbox annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    # Horizontal GTBox, (x1,y1,x2,y2)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    # Horizontal GTBox, (x,y,w,h,theta)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=1, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
