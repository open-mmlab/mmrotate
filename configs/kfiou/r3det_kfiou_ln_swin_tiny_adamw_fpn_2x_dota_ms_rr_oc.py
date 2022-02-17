_base_ = ['./r3det_kfiou_ln_swin_tiny_adamw_fpn_1x_dota_ms_rr_oc.py']

evaluation = dict(interval=24, metric='mAP')
runner = dict(type='EpochBasedRunner', max_epochs=24)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[18, 22])
