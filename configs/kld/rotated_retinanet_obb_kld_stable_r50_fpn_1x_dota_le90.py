_base_ = ['../rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py']

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=5.5)))
