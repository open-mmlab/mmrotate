# Rotated FCOS

> [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143882011-45b234bc-d04b-4bbe-a822-94bec057ac86.png"/>
</div>

We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction
fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3,
and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well
as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation
related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all
hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the
only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7% in AP with single-model
and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler. For the
first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We
hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks.

## Results and Models

DOTA1.0

|         Backbone         |  mAP  | Angle | Separate Angle | Tricks | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                     Configs                                                     |                                                                                                                                                                                   Download                                                                                                                                                                                   |
| :----------------------: | :---: | :---: | :------------: | :----: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 70.70 | le90  |       Y        |   Y    |   1x    |   4.18   |      26.4      |  -  |     2      |              [rotated-fcos-hbox-le90_r50_fpn_1x_dota](./rotated-fcos-hbox-le90_r50_fpn_1x_dota.py)              |       [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90_20220409_023250.log.json)       |
| ResNet50 (1024,1024,200) | 71.28 | le90  |       N        |   Y    |   1x    |   4.18   |      25.9      |  -  |     2      |                   [rotated-fcos-le90_r50_fpn_1x_dota](./rotated-fcos-le90_r50_fpn_1x_dota.py)                   |                           [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90/rotated_fcos_r50_fpn_1x_dota_le90-d87568ed.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90/rotated_fcos_r50_fpn_1x_dota_le90_20220413_163526.log.json)                           |
| ResNet50 (1024,1024,200) | 71.76 | le90  |       Y        |   Y    |   1x    |   4.23   |      25.7      |  -  |     2      | [rotated-fcos-hbox-le90_r50_fpn_csl-gaussian_1x_dota](./rotated-fcos-hbox-le90_r50_fpn_csl-gaussian_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90-4e044ad2.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90_20220409_080616.log.json) |
| ResNet50 (1024,1024,200) | 71.89 | le90  |       N        |   Y    |   1x    |   4.18   |      26.2      |  -  |     2      |               [rotated-fcos-le90_r50_fpn_kld_1x_dota](./rotated-fcos-le90_r50_fpn_kld_1x_dota.py)               |                   [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_kld_r50_fpn_1x_dota_le90/rotated_fcos_kld_r50_fpn_1x_dota_le90-ecafdb2b.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_kld_r50_fpn_1x_dota_le90/rotated_fcos_kld_r50_fpn_1x_dota_le90_20220409_202939.log.json)                   |

**Notes:**

- `MS` means multiple scale image split.
- `RR` means random rotation.
- `Rotated IoU Loss` need mmcv version 1.5.0 or above.
- `Separate Angle` means angle loss is calculated separately.
  At this time bbox loss uses horizontal bbox loss such as `IoULoss`, `GIoULoss`.
- Tricks means setting `norm_on_bbox`, `centerness_on_reg`, `center_sampling` as `True`.
- Inf time was tested on a single RTX3090.

## Citation

```
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:1904.01355},
  year={2019}
}
```
