_base_ = './redet_re50_refpn_1x_dota_le90.py'

optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
