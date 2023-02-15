# CSL

> [Arbitrary-Oriented Object Detection with Circular Smooth Label](https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/csl.jpg" width="800"/>
</div>

Arbitrary-oriented object detection has recently attracted increasing attention in vision for their importance
in aerial imagery, scene text, and face etc. In this paper, we show that existing regression-based rotation detectors
suffer the problem of discontinuous boundaries, which is directly caused by angular periodicity or corner ordering.
By a careful study, we find the root cause is that the ideal predictions are beyond the defined range. We design a
new rotation detection baseline, to address the boundary problem by transforming angular prediction from a regression
problem to a classification task with little accuracy loss, whereby high-precision angle classification is devised in
contrast to previous works using coarse-granularity in rotation detection. We also propose a circular smooth label (CSL)
technique to handle the periodicity of the angle and increase the error tolerance to adjacent angles. We further
introduce four window functions in CSL and explore the effect of different window radius sizes on detection performance.
Extensive experiments and visual analysis on two large-scale public datasets for aerial images i.e. DOTA, HRSC2016,
as well as scene text dataset ICDAR2015 and MLT, show the effectiveness of our approach.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | Window func. | Omega | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                              Configs                                                              |                                                                                                                                                                                                      Download                                                                                                                                                                                                      |
| :----------------------: | :---: | :---: | :----------: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 68.42 | le90  |      -       |   -   |   1x    |   3.38   |      17.8      |  -  |     2      |        [rotated-retinanet-rbox-le90_r50_fpn_1x_dota](../rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py)         |                       [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90_20220128_130740.log.json)                       |
| ResNet50 (1024,1024,200) | 68.79 | le90  |      -       |   -   |   1x    |   2.36   |      25.9      |  -  |     2      |    [rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota](../rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota.py)     |             [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90-01de71b5.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90_20220303_183714.log.json)             |
| ResNet50 (1024,1024,200) | 69.51 | le90  |   Gaussian   |   4   |   1x    |   2.60   |      24.0      |  -  |     2      | [rotated-retinanet-rbox-le90_r50_fpn_csl-gaussian_amp-1x_dota](./rotated-retinanet-rbox-le90_r50_fpn_csl-gaussian_amp-1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/csl/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90-b4271aed.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/csl/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90_20220321_010033.log.json) |

## Citation

```
@inproceedings{yang2020arbitrary,
    title={Arbitrary-Oriented Object Detection with Circular Smooth Label},
    author={Yang, Xue and Yan, Junchi},
    booktitle={European Conference on Computer Vision},
    pages={677--694},
    year={2020}
}
```
