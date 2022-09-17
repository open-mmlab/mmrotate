_base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90.py']

optim_wrapper = dict(type='AmpOptimWrapper')
