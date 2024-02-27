# H2RBox-v2

> [H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection](https://arxiv.org/pdf/2304.04403)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/h2rbox_v2.png" width="800"/>
</div>

With the rapidly increasing demand for oriented object detection, e.g. in autonomous driving and remote sensing, the recently proposed paradigm involving weakly-supervised detector H2RBox for learning rotated box (RBox) from the more readily-available horizontal box (HBox) has shown promise. This paper presents H2RBox-v2, to further bridge the gap between HBox-supervised and RBox-supervised oriented object detection. Specifically, we propose to leverage the reflection symmetry via flip and rotate consistencies, using a weakly-supervised network branch similar to H2RBox, together with a novel self-supervised branch that learns orientations from the symmetry inherent in visual objects. The detector is further stabilized and enhanced by practical techniques to cope with peripheral issues e.g. angular periodicity. To our best knowledge, H2RBox-v2 is the first symmetry-aware self-supervised paradigm for oriented object detection. In particular, our method shows less susceptibility to low-quality annotation and insufficient training data compared to H2RBox. Specifically, H2RBox-v2 achieves very close performance to a rotation annotation trained counterpart -- Rotated FCOS: 1) DOTA-v1.0/1.5/2.0: 72.31%/64.76%/50.33% vs. 72.44%/64.53%/51.77%; 2) HRSC: 89.66% vs. 88.99%; 3) FAIR1M: 42.27% vs. 41.25%.

## Results and models

DOTA1.0

|         Backbone         | AP50  | lr schd | Mem (GB) | Inf Time (fps) |  Aug  | Batch Size |                                      Configs                                      |                                                                                                                                                        Download                                                                                                                                                        |
| :----------------------: | :---: | :-----: | :------: | :------------: | :---: | :--------: | :-------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 72.59 |   1x    |  10.10   |      29.1      |   -   |     2      |       [h2rbox_v2-le90_r50_fpn-1x_dota](./h2rbox_v2-le90_r50_fpn-1x_dota.py)       |            [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_dota/h2rbox_v2-le90_r50_fpn-1x_dota-fa5ad1d2.pth)   \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_dota/h2rbox_v2-le90_r50_fpn-1x_dota-20230313_103051.json)            |
| ResNet50 (1024,1024,200) | 78.25 |   1x    |  10.33   |      29.1      | MS+RR |     2      | [h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota](./h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota/h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-5e0e53e1.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota/h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-20230324_011934.json) |

DOTA1.5

|         Backbone         | AP50  | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                   Configs                                   |                                                                                                                                                  Download                                                                                                                                                  |
| :----------------------: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 64.76 |   1x    |  10.95   |      29.1      |  -  |     2      | [h2rbox_v2-le90_r50_fpn-1x_dotav15](./h2rbox_v2-le90_r50_fpn-1x_dotav15.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_dotav15/h2rbox_v2-le90_r50_fpn-1x_dotav15-3adc0309.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_dotav15/h2rbox_v2-le90_r50_fpn-1x_dotav15-20230316_192940.json) |

DOTA2.0

|         Backbone         | AP50  | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                  Configs                                  |                                                                                                                                                Download                                                                                                                                                |
| :----------------------: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-----------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 50.33 |   1x    |  11.02   |      29.1      |  -  |     2      | [h2rbox_v2-le90_r50_fpn-1x_dotav2](./h2rbox_v2-le90_r50_fpn-1x_dotav2.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_dotav2/h2rbox_v2-le90_r50_fpn-1x_dotav2-b1ec4d3c.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_dotav2/h2rbox_v2-le90_r50_fpn-1x_dotav2-20230316_200353.json) |

HRSC

|         Backbone         | AP50  | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                   Configs                                   |                                                                                                                                                  Download                                                                                                                                                   |
| :----------------------: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 89.66 |   1x    |   5.50   |      45.9      |  -  |     2      |    [h2rbox_v2-le90_r50_fpn-6x_hrsc](./h2rbox_v2-le90_r50_fpn-6x_hrsc.py)    |       [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn-6x_hrsc/h2rbox_v2-le90_r50_fpn-6x_hrsc-b3b5e06b.pth)  \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn-6x_hrsc/h2rbox_v2-le90_r50_fpn-6x_hrsc-20230312_073744.json)       |
| ResNet50 (1024,1024,200) | 89.56 |   1x    |   5.50   |      45.9      | RR  |     2      | [h2rbox_v2-le90_r50_fpn_rr-6x_hrsc](./h2rbox_v2-le90_r50_fpn_rr-6x_hrsc.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn_rr-6x_hrsc/h2rbox_v2-le90_r50_fpn_rr-6x_hrsc-ee6e851a.pth)  \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/h2rbox_v2-le90_r50_fpn_rr-6x_hrsc/h2rbox_v2-le90_r50_fpn_rr-6x_hrsc-20230312_073800.json) |

## Citation

```
@inproceedings{yu2023h2rboxv2,
title={H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection}, 
author={Yi Yu and Xue Yang and Qingyun Li and Yue Zhou and Gefan Zhang and Feipeng Da and Junchi Yan},
year={2023},
booktitle={Advances in Neural Information Processing Systems}
}
```
