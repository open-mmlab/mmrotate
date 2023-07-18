# H2RBox

> [H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection](https://arxiv.org/abs/2210.06742)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/yangxue0827/h2rbox-mmrotate/blob/main/configs/h2rbox/pipeline.png" width="800"/>
</div>

Oriented object detection emerges in many applications from aerial images to autonomous driving, while many existing detection benchmarks are annotated with horizontal bounding box only which is also less costive than fine-grained rotated box, leading to a gap between the readily available training corpus and the rising demand for oriented object detection.  This paper proposes a simple yet effective oriented object detection approach called H2RBox merely using horizontal box annotation for weakly-supervised training, which closes the above gap and shows competitive performance even against those trained with rotated boxes.  The cores of our method are weakly- and self-supervised learning, which predicts the angle of the object by learning the consistency of two different views. To our best knowledge, H2RBox is the first horizontal box annotation-based oriented object detector. Compared to an alternative i.e. horizontal box-supervised instance segmentation with our post adaption to oriented object detection, our approach is not susceptible to the prediction quality of mask and can perform more robustly in complex scenes containing a large number of dense objects and outliers. Experimental results show that H2RBox has significant performance and speed advantages over horizontal box-supervised instance segmentation methods, as well as lower memory requirements. While compared to rotated box-supervised oriented object detectors, our method shows very close performance and speed, and even surpasses them in some cases.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | MS  | Batch Size |                                      Configs                                      |                                                                                                                                                     Download                                                                                                                                                     |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 68.75 | le90  |   1x    |   6.25   |                |  -  |     2      |    [h2rbox-le90_r50_fpn_adamw-1x_dota](./h2rbox-le90_r50_fpn_adamw-1x_dota.py)    |       [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota/h2rbox-le90_r50_fpn_adamw-1x_dota-d02c933a.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota/h2rbox-le90_r50_fpn_adamw-1x_dota-20221124_153420.json)       |
| ResNet50 (1024,1024,200) | 71.31 | le90  |   3x    |   6.64   |                |  -  |     2      |    [h2rbox-le90_r50_fpn_adamw-3x_dota](./h2rbox-le90_r50_fpn_adamw-3x_dota.py)    |       [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-3x_dota/h2rbox-le90_r50_fpn_adamw-3x_dota-8bca2d7f.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-3x_dota/h2rbox-le90_r50_fpn_adamw-3x_dota-20221124_180458.json)       |
| ResNet50 (1024,1024,200) | 74.43 | le90  |   1x    |   6.12   |       -        |  √  |     2      | [h2rbox-le90_r50_fpn_adamw-1x_dota-ms](./h2rbox-le90_r50_fpn_adamw-1x_dota-ms.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota-ms/h2rbox-le90_r50_fpn_adamw-1x_dota-ms-30dcdc68.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dota-ms/h2rbox-le90_r50_fpn_adamw-1x_dota-ms-20221124_224240.json) |

DOTA1.5

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | MS  | Batch Size |                                          Configs                                          |                                                                                                                                                     Download                                                                                                                                                     |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :---------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 60.33 | le90  |   1x    |   6.83   |                |  -  |     2      | [h2rbox-le90_r50_fpn_adamw-1x_dotav15](./dotav15/h2rbox-le90_r50_fpn_adamw-1x_dotav15.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dotav15/h2rbox-le90_r50_fpn_adamw-1x_dotav15-5f2178e6.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dotav15/h2rbox-le90_r50_fpn_adamw-1x_dotav15-20221125_173828.json) |
| ResNet50 (1024,1024,200) | 62.65 | le90  |   1x    |   6.34   |                |  -  |     2      | [h2rbox-le90_r50_fpn_adamw-3x_dotav15](./dotav15/h2rbox-le90_r50_fpn_adamw-3x_dotav15.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-3x_dotav15/h2rbox-le90_r50_fpn_adamw-3x_dotav15-d8337c8e.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-3x_dotav15/h2rbox-le90_r50_fpn_adamw-3x_dotav15-20221126_135218.json) |

DOTA2.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | MS  | Batch Size |                                        Configs                                         |                                                                                                                                                   Download                                                                                                                                                   |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 47.08 | le90  |   1x    |   7.58   |                |  -  |     2      | [h2rbox-le90_r50_fpn_adamw-1x_dotav2](./dotav2/h2rbox-le90_r50_fpn_adamw-1x_dotav2.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dotav2/h2rbox-le90_r50_fpn_adamw-1x_dotav2-366ced3d.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dotav2/h2rbox-le90_r50_fpn_adamw-1x_dotav2-20221126_141507.json) |
| ResNet50 (1024,1024,200) | 50.20 | le90  |   1x    |   7.74   |                |  -  |     2      | [h2rbox-le90_r50_fpn_adamw-3x_dotav2](./dotav2/h2rbox-le90_r50_fpn_adamw-3x_dotav2.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-3x_dotav2/h2rbox-le90_r50_fpn_adamw-3x_dotav2-85bf9bfa.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-3x_dotav2/h2rbox-le90_r50_fpn_adamw-3x_dotav2-20221126_225316.json) |

DIOR

|      Backbone      | AP50:95 | AP50  | AP75  | Angle | lr schd | Mem (GB) | Inf Time (fps) | MS  | Batch Size |                                     Configs                                      |                                                                                                                                               Download                                                                                                                                               |
| :----------------: | :-----: | :---: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800) |  33.06  | 57.40 | 32.50 | le90  |   1x    |   3.83   |                |  -  |     2      | [h2rbox-le90_r50_fpn_adamw-1x_dior](./dior/h2rbox-le90_r50_fpn_adamw-1x_dior.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dior/h2rbox-le90_r50_fpn_adamw-1x_dior-949b0e4c.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/h2rbox/h2rbox-le90_r50_fpn_adamw-1x_dior/h2rbox-le90_r50_fpn_adamw-1x_dior-20221130_204038.json) |

**Notes:**

- `MS` means multiple scale image split.
- `Inf Time` was tested on a single RTX3090.
- [Original PyTorch Implementation (Paper) for H2RBox](https://github.com/yangxue0827/h2rbox-mmrotate)
- [Jittor Implementation for H2RBox](https://github.com/yangxue0827/h2rbox-jittor)
- [JDet Implementation for H2RBox](https://github.com/Jittor/JDet)
- The memory log of MMRotate 1.x is dynamic, so the last iteration is recorded here.

## Citation

```
@article{yang2023h2rbox,
  title={H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection},
  author={Yang, Xue and Zhang, Gefan and Li, Wentong and Wang, Xuehui and Zhou, Yue and Yan, Junchi},
	booktitle={International Conference on Learning Representations},
	year={2023}
}

```
