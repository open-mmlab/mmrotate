_base_ = './s2anet-le135_r50_fpn_1x_dota.py'

optim_wrapper = dict(type='AmpOptimWrapper')
