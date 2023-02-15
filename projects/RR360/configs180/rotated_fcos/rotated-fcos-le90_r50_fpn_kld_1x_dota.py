_base_ = 'rotated-fcos-le90_r50_fpn_1x_dota.py'

model = dict(
    bbox_head=dict(
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1,
            loss_weight=1.0)))
