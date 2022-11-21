_base_ = ['../../../configs/r3det/r3det-oc_r50_fpn_1x_dota.py']

custom_imports = dict(imports=['projects.example_project.dummy'])

_base_.model.backbone.type = 'DummyResNet'
