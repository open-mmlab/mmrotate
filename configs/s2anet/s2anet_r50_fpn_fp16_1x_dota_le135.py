_base_ = [
    './s2anet_r50_fpn_1x_dota_le135.py'
]

fp16 = dict(loss_scale='dynamic')
model = dict(train_cfg=dict(rpn=dict(assigner=dict(gpu_assign_thr=200))))

