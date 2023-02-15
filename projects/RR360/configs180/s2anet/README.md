# S2ANet

> [Align Deep Features for Oriented Object Detection](https://ieeexplore.ieee.org/document/9377550)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/s2a.png" width="800"/>
</div>

The past decade has witnessed significant progress on detecting objects in aerial images that are often distributed with large-scale variations and arbitrary orientations. However, most of existing methods rely on heuristically defined anchors with different scales, angles, and aspect ratios, and usually suffer from severe misalignment between anchor boxes (ABs) and axis-aligned convolutional features, which lead to the common inconsistency between the classification score and localization accuracy. To address this issue, we propose a single-shot alignment network (S²A-Net) consisting of two modules: a feature alignment module (FAM) and an oriented detection module (ODM). The FAM can generate high-quality anchors with an anchor refinement network and adaptively align the convolutional features according to the ABs with a novel alignment convolution. The ODM first adopts active rotating filters to encode the orientation information and then produces orientation-sensitive and orientation-invariant features to alleviate the inconsistency between classification score and localization accuracy. Besides, we further explore the approach to detect objects in large-size images, which leads to a better trade-off between speed and accuracy. Extensive experiments demonstrate that our method can achieve the state-of-the-art performance on two commonly used aerial objects' data sets (i.e., DOTA and HRSC2016) while keeping high efficiency.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                       Configs                                                        |                                                                                                                                                                                  Download                                                                                                                                                                                  |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 69.79 | le135 |   1x    |   3.38   |      17.2      |  -  |     2      | [rotated-retinanet-rbox-le135_r50_fpn_1x_dota](../rotated_retinanet/rotated-retinanet-rbox-le135_r50_fpn_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135_20220128_130755.log.json) |
| ResNet50 (1024,1024,200) | 73.91 | le135 |   1x    |   3.14   |      15.5      |  -  |     2      |                          [s2anet-le135_r50_fpn_1x_dota](./s2anet-le135_r50_fpn_1x_dota.py)                           |                                          [model](https://download.openmmlab.com/mmrotate/v0.1.0/s2anet/s2anet_r50_fpn_1x_dota_le135/s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/s2anet/s2anet_r50_fpn_1x_dota_le135/s2anet_r50_fpn_1x_dota_le135_20220124_163529.log.json)                                          |
| ResNet50 (1024,1024,200) | 74.19 | le135 |   1x    |   2.17   |      17.4      |  -  |     2      |                      [s2anet-le135_r50_fpn_amp-1x_dota](./s2anet-le135_r50_fpn_amp-1x_dota.py)                       |                                [model](https://download.openmmlab.com/mmrotate/v0.1.0/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135/s2anet_r50_fpn_fp16_1x_dota_le135-5cac515c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135/s2anet_r50_fpn_fp16_1x_dota_le135_20220303_194910.log.json)                                |

HRSC

|      Backbone      |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                              Configs                              | Download |
| :----------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :---------------------------------------------------------------: | :------: |
| ResNet50 (800,800) | 89.75 | le90  |   3x    |          |                |  -  |     2      | [s2anet_r50_fpn_3x_hrsc_le135](./s2anet_r50_fpn_3x_hrsc_le135.py) |          |

## Citation

```
@article{han2021align,
  title={Align deep features for oriented object detection},
  author={Han, Jiaming and Ding, Jian and Li, Jie and Xia, Gui-Song},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
```
