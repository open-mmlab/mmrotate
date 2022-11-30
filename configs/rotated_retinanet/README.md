# Rotated RetinaNet

> [Focal loss for dense object detection](https://arxiv.org/pdf/1708.02002.pdf)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/retina.png" width="800"/>
</div>

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors.

## Results and Models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) |  Aug  | Batch Size |                                                   Configs                                                   |                                                                                                                                                                                            Download                                                                                                                                                                                            |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :---: | :--------: | :---------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 64.55 |  oc   |   1x    |   3.38   |      15.7      |   -   |     2      |         [rotated-retinanet-hbox-oc_r50_fpn_1x_dota](./rotated-retinanet-hbox-oc_r50_fpn_1x_dota.py)         |                 [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc_20220121_095315.log.json)                 |
| ResNet50 (1024,1024,200) | 68.42 | le90  |   1x    |   3.38   |      16.9      |   -   |     2      |       [rotated-retinanet-rbox-le90_r50_fpn_1x_dota](./rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py)       |             [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90_20220128_130740.log.json)             |
| ResNet50 (1024,1024,200) | 68.79 | le90  |   1x    |   2.36   |      22.4      |   -   |     2      |   [rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota](./rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota.py)   |   [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90-01de71b5.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90_20220303_183714.log.json)   |
| ResNet50 (1024,1024,200) | 69.79 | le135 |   1x    |   3.38   |      17.2      |   -   |     2      |      [rotated-retinanet-rbox-le135_r50_fpn_1x_dota](./rotated-retinanet-rbox-le135_r50_fpn_1x_dota.py)      |           [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135_20220128_130755.log.json)           |
| ResNet50 (1024,1024,500) | 76.50 | le90  |   1x    |          |      17.5      | MS+RR |     2      | [rotated-retinanet-rbox-le90_r50_fpn_rr-1x_dota-ms](./rotated-retinanet-rbox-le90_r50_fpn_rr-1x_dota-ms.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90_20220210_114843.log.json) |

HRSC

|      Backbone      |  mAP  | AP50  | AP75  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                Configs                                                |                                                                                                                                                                                      Download                                                                                                                                                                                      |
| :----------------: | :---: | :---: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :---------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,512) | 45.09 | 79.30 | 46.90 |  oc   |   6x    |   1.56   |      39.2      | RR  |     2      |   [rotated-retinanet-hbox-oc_r50_fpn_rr-6x_hrsc](./rotated-retinanet-hbox-oc_r50_fpn_rr-6x_hrsc.py)   |     [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc/rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc-f37eada6.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc/rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc_20220412_103639.log.json)     |
| ResNet50 (800,512) | 52.06 | 84.80 | 58.10 | le90  |   6x    |   1.56   |      38.2      | RR  |     2      | [rotated-retinanet-rbox-le90_r50_fpn_rr-6x_hrsc](./rotated-retinanet-rbox-le90_r50_fpn_rr-6x_hrsc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90-ee4f18af.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90_20220412_110739.log.json) |

DIOR

|      Backbone      |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                             Configs                                             |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :---------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800) | 50.26 |  oc   |   1x    |   3.66   |                |  -  |     2      |   [rotated-retinanet-rbox-oc_r50_fpn_1x_dior](./rotated-retinanet-rbox-oc_r50_fpn_1x_dior.py)   |     [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_retinanet/rotated-retinanet-rbox-oc_r50_fpn_1x_dior/rotated-retinanet-rbox-oc_r50_fpn_1x_dior-dbdbc2f8.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_retinanet/rotated-retinanet-rbox-oc_r50_fpn_1x_dior/rotated-retinanet-rbox-oc_r50_fpn_1x_dior_20221124_173852.json)     |
| ResNet50 (800,800) | 53.94 | le90  |   1x    |   3.62   |                |  -  |     2      | [rotated-retinanet-rbox-le90_r50_fpn_1x_dior](./rotated-retinanet-rbox-le90_r50_fpn_1x_dior.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dior/rotated-retinanet-rbox-le90_r50_fpn_1x_dior-caf9143c.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dior/rotated-retinanet-rbox-le90_r50_fpn_1x_dior_20221124_230602.json) |

Notes:

- `MS` means multiple scale image split.
- `RR` means random rotation.
- `hbb` means the input of the assigner is the predicted box and the horizontal box that can surround the GT. `obb` means the input of the assigner is the predicted box and the GT. They can be switched by `assign_by_circumhbbox`  in `RotatedRetinaHead`.

## Citation

```
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
