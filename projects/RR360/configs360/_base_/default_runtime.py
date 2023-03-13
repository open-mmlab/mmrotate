default_scope = 'mmrotate'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,
        save_best=['dota/AP50'],
        rule='greater',
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='WandbVisBackend',
    #      init_kwargs=dict(project='trbox'))
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = dict(
    type='RR360LocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

custom_imports = dict(
    imports=[
        # 'mmcls.models',
        'projects.RR360.visualization',
        'projects.RR360.structures',
        'projects.RR360.datasets.transforms',
        'projects.RR360.evaluation',
        # 'projects.RR360.models'
    ],
    allow_failed_imports=False)
