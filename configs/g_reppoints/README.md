# G-Rep: Gaussian Representation for Arbitrary-Oriented Object Detection.

<!-- [ALGORITHM] -->
## Abstract

Core code will release later.

## Results and models

### DOTA1.0

#### RepPoints
|    Backbone   |    mAP   | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | Configs | Download |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------------:|
| ResNet50 (1024,1024,200) | 59.44 | oc | 1x | 3.45 | 15.9 | - | 2 | [rotated_reppoints_r50_fpn_1x_dota_oc](../rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py) |  [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc-d38ce217.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc_20220205_145010.log.json)
| ResNet50 (1024,1024,200) | 69.49 | le135 | 1x | 4.05 | 10.5 | - | 2 | [g_reppoints_r50_fpn_1x_dota_le135](./g_reppoints_r50_fpn_1x_dota_le135.py) |  [model](https://download.openmmlab.com/mmrotate/v0.1.0/g_reppoints/g_reppoints_r50_fpn_1x_dota_le135/g_reppoints_r50_fpn_1x_dota_le135-b840eed7.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/g_reppoints/g_reppoints_r50_fpn_1x_dota_le135/g_reppoints_r50_fpn_1x_dota_le135_20220202_233631.log.json)


## Citation
Coming soon!
