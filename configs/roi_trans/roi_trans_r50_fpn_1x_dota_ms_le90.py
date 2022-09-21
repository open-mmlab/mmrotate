_base_ = './roi_trans_r50_fpn_1x_dota_le90.py'

data_root = 'data/split_ms_dota/'

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
