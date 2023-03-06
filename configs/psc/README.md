# PSC

> [Phase-Shifting Coder: Predicting Accurate Orientation in Oriented Object Detection](https://arxiv.org/abs/2211.06368)

<!-- [ALGORITHM] -->

## Abstract

With the vigorous development of computer vision, oriented object detection has gradually been featured. In this paper, a novel differentiable angle coder named phase-shifting coder (PSC) is proposed to accurately predict the orientation of objects, along with a dual-frequency version PSCD. By mapping rotational periodicity of different cycles into phase of different frequencies, we provide a unified framework for various periodic fuzzy problems in oriented object detection. Upon such framework, common problems in oriented object detection such as boundary discontinuity and square-like problems are elegantly solved in a unified form. Visual analysis and experiments on three datasets prove the effectiveness and the potentiality of our approach. When facing scenarios requiring high-quality bounding boxes, the proposed methods are expected to give a competitive performance.

## Results and models

DOTA1.0

|         Backbone         | AP50  | AP75  | Angle | Dual-freqency | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                              Configs                                                              |                                                                                                                                                                                          Download                                                                                                                                                                                          |
| :----------------------: | :---: | :---: | :---: | :-----------: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 71.14 | 40.28 | le90  |     True      |   1x    |   4.29   |      21.9      |  -  |     2      |              [rotated-fcos-hbox-le90_r50_fpn_psc-dual_1x_dota](./rotated-fcos-hbox-le90_r50_fpn_psc-dual_1x_dota.py)              |                   [model](https://download.openmmlab.com/mmrotate/v1.0/psc/rotated-fcos-hbox-le90_r50_fpn_psc-dual_1x_dota/rotated-fcos-hbox-le90_r50_fpn_psc-dual_1x_dota-326e276b.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/psc/rotated-fcos-hbox-le90_r50_fpn_psc-dual_1x_dota/rotated-fcos-hbox-le90_r50_fpn_psc-dual_1x_dota-20221115_233457.json)                   |
| ResNet50 (1024,1024,200) | 71.92 | 45.84 | le90  |     True      |   1x    |   2.26   |      20.1      |  -  |     2      | [rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota (note1)](./rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/psc/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota-951713be.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/psc/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota-20221121_131433.json) |

note1: Trained with --cfg-options load_from=rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth

HRSC

|      Backbone      | AP50  | AP75  | Angle | Dual-freqency | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                    Configs                                                    |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------: | :---: | :---: | :---: | :-----------: | :-----: | :------: | :------------: | :-: | :--------: | :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,512) | 90.10 | 79.39 | le90  |       -       |   6x    |   1.46   |      37.0      | RR  |     2      |      [rotated-fcos-hbox-le90_r50_fpn_psc_rr-6x_hrsc](./rotated-fcos-hbox-le90_r50_fpn_psc_rr-6x_hrsc.py)      |           [model](https://download.openmmlab.com/mmrotate/v1.0/psc/rotated-fcos-hbox-le90_r50_fpn_psc_rr-6x_hrsc/rotated-fcos-hbox-le90_r50_fpn_psc_rr-6x_hrsc-3da09c7a.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/psc/rotated-fcos-hbox-le90_r50_fpn_psc_rr-6x_hrsc/rotated-fcos-hbox-le90_r50_fpn_psc_rr-6x_hrsc-20221114_193627.json)           |
| ResNet50 (800,512) | 84.86 | 60.07 | le90  |       -       |   6x    |   1.44   |      35.7      | RR  |     2      | [rotated-retinanet-rbox-le90_r50_fpn_psc_rr-6x_hrsc](./rotated-retinanet-rbox-le90_r50_fpn_psc_rr-6x_hrsc.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/psc/rotated-retinanet-rbox-le90_r50_fpn_psc_rr-6x_hrsc/rotated-retinanet-rbox-le90_r50_fpn_psc_rr-6x_hrsc-d2e78a2d.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/psc/rotated-retinanet-rbox-le90_r50_fpn_psc_rr-6x_hrsc/rotated-retinanet-rbox-le90_r50_fpn_psc_rr-6x_hrsc-20221119_190110.json) |

## Citation

```
@inproceedings{yu2023psc,
    author = {Yu, Yi and Da, Feipeng},
    title = {Phase-Shifting Coder: Predicting Accurate Orientation in Oriented Object Detection},
    booktitle = {Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2023},
    url = {https://arxiv.org/abs/2211.06368}
}
```
