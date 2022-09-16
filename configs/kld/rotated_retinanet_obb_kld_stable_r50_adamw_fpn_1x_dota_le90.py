_base_ = './rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90.py'

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))
