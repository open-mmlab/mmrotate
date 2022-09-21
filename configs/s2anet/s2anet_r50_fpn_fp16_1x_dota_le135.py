_base_ = ['./s2anet_r50_fpn_1x_dota_le135.py']

optim_wrapper = dict(type='AmpOptimWrapper')
