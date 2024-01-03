# Point2RBox

> [Point2RBox: Combine Knowledge from Synthetic Visual Patterns for End-to-end Oriented Object Detection with Single Point Supervision](https://arxiv.org/pdf/2311.14758)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/point2rbox.png" width="800"/>
</div>

With the rapidly increasing demand for oriented object detection (OOD), recent research involving weakly-supervised detectors for learning rotated box (RBox) from the horizontal box (HBox) has attracted more and more attention. In this paper, we explore a more challenging yet label-efficient setting, namely single point-supervised OOD, and present our approach called Point2RBox. Specifically, we propose to leverage two principles: 1) Synthetic pattern knowledge combination: By sampling around each labelled point on the image, we transfer the object feature to synthetic visual patterns with the known bounding box to provide the knowledge for box regression. 2) Transform self-supervision: With a transformed input image (e.g. scaled/rotated), the output RBoxes are trained to follow the same transformation so that the network can perceive the relative size/rotation between objects. The detector is further enhanced by a few devised techniques to cope with peripheral issues, e.g. the anchor/layer assignment as the size of the object is not available in our point supervision setting. To our best knowledge, Point2RBox is the first end-to-end solution for point-supervised OOD. In particular, our method uses a lightweight paradigm, yet it achieves a competitive performance among point-supervised alternatives, 41.05%/27.62%/80.01% on DOTA/DIOR/HRSC datasets.

## Basic patterns

Extract [basic_patterns.zip](https://github.com/yuyi1005/point2rbox/files/13816301/basic_patterns.zip) to data folder. The path can also be modified in config files.

## Results and models

DOTA1.0

|         Backbone         | AP50  | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                       Configs                       |                                                                                                                    Download                                                                                                                    |
| :----------------------: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 41.87 |   1x    |  16.12   |     111.7      |  -  |     2      | [point2rbox-yolof-dota](./point2rbox-yolof-dota.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/point2rbox/point2rbox-yolof-dota/point2rbox-yolof-dota-c94da82d.pth)   \| [log](https://download.openmmlab.com/mmrotate/v1.0/point2rbox/point2rbox-yolof-dota/point2rbox-yolof-dota.json) |

DIOR

|      Backbone      | AP50  | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                       Configs                       |                                                                                                                    Download                                                                                                                    |
| :----------------: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800) | 27.34 |   1x    |  10.38   |     127.3      |  -  |     2      | [point2rbox-yolof-dior](./point2rbox-yolof-dior.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/point2rbox/point2rbox-yolof-dior/point2rbox-yolof-dior-f4f724df.pth)   \| [log](https://download.openmmlab.com/mmrotate/v1.0/point2rbox/point2rbox-yolof-dior/point2rbox-yolof-dior.json) |

HRSC

|      Backbone      | AP50  | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                       Configs                       |                                                                                                                   Download                                                                                                                    |
| :----------------: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800) | 79.40 |   6x    |   9.60   |     136.9      |  -  |     2      | [point2rbox-yolof-hrsc](./point2rbox-yolof-hrsc.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/point2rbox/point2rbox-yolof-hrsc/point2rbox-yolof-hrsc-9d096323.pth)  \| [log](https://download.openmmlab.com/mmrotate/v1.0/point2rbox/point2rbox-yolof-hrsc/point2rbox-yolof-hrsc.json) |

## Citation

```
@misc{yu2023point2rbox,
title={Point2RBox: Combine Knowledge from Synthetic Visual Patterns for End-to-end Oriented Object Detection with Single Point Supervision},
author={Yi Yu and Xue Yang and Qingyun Li and Feipeng Da and Junchi Yan and Jifeng Dai and Yu Qiao},
year={2023},
eprint={2311.14758},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```
