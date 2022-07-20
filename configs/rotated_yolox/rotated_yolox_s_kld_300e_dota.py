_base_ = './rotated_yolox_s_300e_dota.py'

# model settings
model = dict(
    bbox_head=dict(
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            reduction='sum',
            loss_weight=27.5)))
