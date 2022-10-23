_base_ = './roi-trans-le90_r50_fpn_1x_dota.py'

data_root = 'data/split_ms_dota/'

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
