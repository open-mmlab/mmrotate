_base_ = './rotated_dal_retinanet_obb_r50_fpn_1x_dota_le90.py'

# learning policy
lr_config = dict(step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
