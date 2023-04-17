# H2RBox-v2

> [H2RBox-v2: Boosting HBox-supervised Oriented Object Detection via Symmetric Learning](https://arxiv.org/pdf/2304.04403)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/h2rbox_v2.png" width="800"/>
</div>

With the increasing demand for oriented object detection e.g. in autonomous driving and remote sensing, the oriented annotation has become a labor-intensive work. To make full use of existing horizontally annotated datasets and reduce the annotation cost, a weakly-supervised detector H2RBox for learning the rotated box (RBox) from the horizontal box (HBox) has been proposed and received great attention. This paper presents a new version, H2RBox-v2, to further bridge the gap between HBox-supervised and RBox-supervised oriented object detection. While exploiting axisymmetry via flipping and rotating consistencies is available through our theoretical analysis, H2RBox-v2, using a weakly-supervised branch similar to H2RBox, is embedded with a novel self-supervised branch that learns orientations from the symmetry inherent in the image of objects. Complemented by modules to cope with peripheral issues, e.g. angular periodicity, a stable and effective solution is achieved. To our knowledge, H2RBox-v2 is the first symmetry-supervised paradigm for oriented object detection. Compared to H2RBox, our method is less susceptible to low annotation quality and insufficient training data, which in such cases is expected to give a competitive performance much closer to fully-supervised oriented object detectors. Specifically, the performance comparison between H2RBox-v2 and Rotated FCOS on DOTA-v1.0/1.5/2.0 is 72.31%/64.76%/50.33% vs. 72.44%/64.53%/51.77%, 89.66% vs. 88.99% on HRSC, and 42.27% vs. 41.25% on FAIR1M.

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
@misc{yu2023h2rboxv2,
title={H2RBox-v2: Boosting HBox-supervised Oriented Object Detection via Symmetric Learning},
author={Yi Yu and Xue Yang and Qingyun Li and Yue Zhou and Gefan Zhang and Feipeng Da and Junchi Yan},
year={2023},
eprint={2304.04403},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```
