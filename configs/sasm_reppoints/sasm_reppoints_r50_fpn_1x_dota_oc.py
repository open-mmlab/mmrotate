_base_ = ['../rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py']

model = dict(
    bbox_head=dict(
        type='SAMRepPointsHead',
        loss_bbox_init=dict(type='BCConvexGIoULoss', loss_weight=0.375)),

    # training and testing settings
    train_cfg=dict(
        refine=dict(
            _delete_=True,
            assigner=dict(type='SASAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)))
