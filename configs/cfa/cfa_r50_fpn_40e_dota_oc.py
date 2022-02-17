_base_ = ['./cfa_r50_fpn_1x_dota_oc.py']

# evaluation
evaluation = dict(interval=40, metric='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 32, 38])
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=10)
