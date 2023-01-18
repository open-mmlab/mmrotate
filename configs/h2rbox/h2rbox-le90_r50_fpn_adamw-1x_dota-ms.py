_base_ = './h2rbox-le90_r50_fpn_adamw-1x_dota.py'
data_root = '/data/nas/dataset_share/DOTA/split_ms_dota1_0/'

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
