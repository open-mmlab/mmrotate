# KFIoU

> [The KFIoU Loss for Rotated Object Detection](https://arxiv.org/pdf/2101.11952.pdf)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/kfiou.png" width="800"/>
</div>

Differing from the well-developed horizontal object detection area whereby the computing-friendly IoU based loss is
readily adopted and well fits with the detection metrics. In contrast, rotation detectors often involve a more
complicated loss based on SkewIoU which is unfriendly to gradient-based training. In this paper, we argue that one
effective alternative is to devise an approximate loss who can achieve trend-level alignment with SkewIoU loss instead
of the strict value-level identity. Specifically, we model the objects as Gaussian distribution and adopt Kalman filter to
inherently mimic the mechanism of SkewIoU by its definition, and show its alignment with the SkewIoU at trend-level. This
is in contrast to recent Gaussian modeling based rotation detectors e.g. GWD, KLD that involves a human-specified
distribution distance metric which requires additional hyperparameter tuning. The resulting new loss called KFIoU is
easier to implement and works better compared with exact SkewIoU, thanks to its full differentiability and ability to
handle the non-overlapping cases. We further extend our technique to the 3-D case which also suffers from the same
issues as 2-D detection. Extensive results on various public datasets (2-D/3-D, aerial/text/face images) with different
base detectors show the effectiveness of our approach.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                                  Download                                                                                                                                                                                  |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 64.55 |  oc   |   1x    |   3.38   |      15.7      |  -  |     2      | [rotated-retinanet-hbox-oc_r50_fpn_1x_dota](../rotated_retinanet/rotated-retinanet-hbox-oc_r50_fpn_1x_dota.py) |       [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc_20220121_095315.log.json)       |
| ResNet50 (1024,1024,200) | 69.60 | le90  |   1x    |   3.38   |      15.1      |  -  |     2      |  [rotated-retinanet-hbox-le90_r50_fpn_kfiou_1x_dota](./rotated-retinanet-hbox-le90_r50_fpn_kfiou_1x_dota.py)   |   [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90-03e02f75.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90_20220209_173225.log.json)   |
| ResNet50 (1024,1024,200) | 69.76 |  oc   |   1x    |   3.39   |      15.6      |  -  |     2      |    [rotated-retinanet-hbox-oc_r50_fpn_kfiou_1x_dota](./rotated-retinanet-hbox-oc_r50_fpn_kfiou_1x_dota.py)     |       [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc-c00be030.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc_20220126_081643.log.json)       |
| ResNet50 (1024,1024,200) | 69.77 | le135 |   1x    |   3.38   |      15.3      |  -  |     2      | [rotated-retinanet-hbox-le135_r50_fpn_kfiou_1x_dota](./rotated-retinanet-hbox-le135_r50_fpn_kfiou_1x_dota.py)  | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135-0eaa4156.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135_20220209_173257.log.json) |

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                   Configs                                   |                                                                                                                                                  Download                                                                                                                                                  |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 69.80 |  oc   |   1x    |   3.54   |      12.4      |  -  |     2      |      [r3det-oc_r50_fpn_1x_dota](../r3det/r3det-oc_r50_fpn_1x_dota.py)       |                   [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc-b1fb045c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc_20220126_191226.log.json)                   |
| ResNet50 (1024,1024,200) | 72.68 |  oc   |   1x    |   3.62   |      12.2      |  -  |     2      | [r3det-oc_r50_fpn_kfiou-ln_1x_dota](./r3det-oc_r50_fpn_kfiou-ln_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc/r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc/r3det_kfiou_ln_r50_fpn_1x_dota_oc_20220123_074507.log.json) |

## Citation

```
@misc{yang2023kfiou,
      title={The KFIoU Loss for Rotated Object Detection},
      author={Xue Yang and Yue Zhou and Gefan Zhang and Jirui Yang and Wentao Wang and Junchi Yan and Xiaopeng Zhang and Qi Tian},
	booktitle={International Conference on Learning Representations},
	year={2023}
}
```
