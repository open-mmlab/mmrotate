# [R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/pdf/1908.05612.pdf)

<!-- [ALGORITHM] -->
## Abstract

![illustration](https://raw.githubusercontent.com/zytx121/image-host/main/imgs/r3det.png)

Rotation detection is a challenging task due to the difficulties of locating the multi-angle objects and separating them effectively from the background. Though considerable progress has been made, for practical settings, there still exist challenges for rotating objects with large aspect ratio, dense distribution and category extremely imbalance. In this paper, we propose an end-to-end refined single-stage rotation detector for fast and accurate object detection by using a progressive regression approach from coarse to fine granularity. Considering the shortcoming of feature misalignment in existing refined single stage detector, we design a feature refinement module to improve detection performance by getting more accurate features. The key idea of feature refinement module is to re-encode the position information of the current refined bounding box to the corresponding feature points through pixel-wise feature interpolation to realize feature reconstruction and alignment. For more accurate rotation estimation, an approximate SkewIoU loss is proposed to solve the problem that the calculation of SkewIoU is not derivable. Experiments on three popular remote sensing public datasets DOTA, HRSC2016, UCAS-AOD as well as one scene text dataset ICDAR2015 show the effectiveness of our approach.

## Results and models

### DOTA1.0

|    Backbone   |    mAP   | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | Configs | Download |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------------:|
| ResNet50 (1024,1024,200) | 64.55 | oc | 1x | 3.38 | 14.8 | - | 2 | [rotated_retinanet_hbb_r50_fpn_1x_dota_oc](../rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py) |  [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc_20220121_095315.log.json)
| ResNet50 (1024,1024,200) | 69.80 | oc | 1x | 3.54 | 12.1 | - | 2 | [r3det_r50_fpn_1x_dota_oc](../r3det/r3det_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc-b1fb045c.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc_20220126_191226.log.json)
| ResNet50 (1024,1024,200) | 70.18 | oc | 1x | 3.23 | 15.1 | - | 2 | [r3det_tiny_r50_fpn_1x_dota_oc](../r3det/r3det_tiny_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_tiny_r50_fpn_1x_dota_oc/r3det_tiny_r50_fpn_1x_dota_oc-c98a616c.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_tiny_r50_fpn_1x_dota_oc/r3det_tiny_r50_fpn_1x_dota_oc_20220209_171624.log.json)


## Citation
```
@inproceedings{yang2021r3det,
    title={R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object},
    author={Yang, Xue and Yan, Junchi and Feng, Ziming and He, Tao},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={35},
    number={4},
    pages={3163--3171},
    year={2021}
}

```
