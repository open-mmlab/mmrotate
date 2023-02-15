_base_ = './h2rbox-le90_r50_fpn_adamw-1x_dota.py'

# training schedule for 3x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=12)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]
