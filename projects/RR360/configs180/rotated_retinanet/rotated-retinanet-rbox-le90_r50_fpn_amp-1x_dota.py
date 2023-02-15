_base_ = ['./rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py']

optim_wrapper = dict(type='AmpOptimWrapper')
