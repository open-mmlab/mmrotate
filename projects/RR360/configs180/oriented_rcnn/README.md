# Oriented R-CNN

> [Oriented R-CNN for Object Detection](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/oriented_rcnn.png" width="800"/>
</div>

Current state-of-the-art two-stage detectors generate oriented proposals through time-consuming schemes. This diminishes the detectors’ speed, thereby becoming the computational bottleneck in advanced oriented object detection systems. This work proposes an effective and simple oriented object detection framework, termed Oriented R-CNN, which is a general two-stage oriented detector with promising accuracy and efficiency. To be specific, in the first stage, we propose an oriented Region Proposal Network (oriented RPN) that directly generates high-quality oriented proposals in a nearly cost-free manner. The second stage is oriented R-CNN head for refining oriented Regions of Interest (oriented RoIs) and recognizing them.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 73.40 | le90  |   1x    |   8.46   |      16.5      |  -  |     2      | [rotated-faster-rcnn-le90_r50_fpn_1x_dota](../rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90_20220131_082156.log.json) |
| ResNet50 (1024,1024,200) | 75.63 | le90  |   1x    |   7.37   |      21.2      |  -  |     2      |             [oriented-rcnn-le90_r50_fpn_amp-1x_dota](./oriented-rcnn-le90_r50_fpn_amp-1x_dota.py)              |         [model](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/oriented_rcnn_r50_fpn_fp16_1x_dota_le90-57c88621.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/oriented_rcnn_r50_fpn_fp16_1x_dota_le90_20220303_195049.log.json)         |
| ResNet50 (1024,1024,200) | 75.69 | le90  |   1x    |   8.46   |      16.2      |  -  |     2      |                 [oriented-rcnn-le90_r50_fpn_1x_dota](./oriented-rcnn-le90_r50_fpn_1x_dota.py)                  |                   [model](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90_20220127_100150.log.json)                   |

## Citation

```
@InProceedings{Xie_2021_ICCV,
  author = {Xie, Xingxing and Cheng, Gong and Wang, Jiabao and Yao, Xiwen and Han, Junwei},
  title = {Oriented R-CNN for Object Detection},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2021},
  pages = {3520-3529} }
```
