_base_ = './rotated-retinanet-rbox-le90_r50_fpn_kld-stable_1x_dota.py'

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))
