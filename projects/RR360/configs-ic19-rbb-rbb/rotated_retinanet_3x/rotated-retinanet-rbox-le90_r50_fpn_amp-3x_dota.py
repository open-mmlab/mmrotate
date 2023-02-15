_base_ = ['./rotated-retinanet-rbox-le90_r50_fpn_3x_dota.py']

optim_wrapper = dict(type='AmpOptimWrapper')
