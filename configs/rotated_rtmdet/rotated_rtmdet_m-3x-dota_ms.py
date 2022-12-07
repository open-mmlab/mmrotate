_base_ = './rotated_rtmdet_l-3x-dota_ms.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.67,
        widen_factor=0.75,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(
        in_channels=192,
        feat_channels=192,
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0)))

# batch_size = (1 GPUs) x (8 samples per GPU) = 8
train_dataloader = dict(batch_size=8, num_workers=8)
