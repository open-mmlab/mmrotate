_base_ = './rotated_yolox_s_300e_dota_le90.py'

# model settings
model = dict(
    bbox_head=dict(
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=True,
            reduction='sum',
            loss_weight=27.5)))
