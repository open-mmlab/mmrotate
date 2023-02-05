# RoI Trans

> [Learning RoI Transformer for Oriented Object Detection in Aerial Images](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/roi_trans.png" width="800"/>
</div>

Object detection in aerial images is an active yet challenging task in computer vision because of the bird’s-eye view perspective, the highly complex backgrounds, and the variant appearances of objects. Especially when detecting densely packed objects in aerial images, methods relying on horizontal proposals for common object detection often introduce mismatches between the Region of Interests (RoIs) and objects. This leads to the common misalignment between the final object classification confidence and localization accuracy. In this paper, we propose a RoI Transformer to address these problems. The core idea of RoI Transformer is to apply spatial transformations on RoIs and learn the transformation parameters under the supervision of oriented bounding box (OBB) annotations. RoI Transformer is with lightweight and can be easily embedded into detectors for oriented object detection. Simply apply the RoI Transformer to light-head RCNN has achieved state-of-the-art performances on two common and challenging aerial datasets, i.e., DOTA and HRSC2016, with a neglectable reduction to detection speed. Our RoI Transformer exceeds the deformable Position Sensitive RoI pooling when oriented bounding-box annotations are available. Extensive experiments have also validated the flexibility and effectiveness of our RoI Transformer

## Results and models

DOTA1.0

|         Backbone          |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) |  Aug  | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :-----------------------: | :---: | :---: | :-----: | :------: | :------------: | :---: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200)  | 73.40 | le90  |   1x    |   8.46   |      16.5      |   -   |     2      | [rotated-faster-rcnn-le90_r50_fpn_1x_dota](../rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90_20220131_082156.log.json) |
| ResNet50 (1024,1024,200)  | 75.75 | le90  |   1x    |   7.56   |      19.3      |   -   |     2      |                 [roi-trans-le90_r50_fpn_amp-1x_dota](./roi-trans-le90_r50_fpn_amp-1x_dota.py)                  |                     [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90/roi_trans_r50_fpn_fp16_1x_dota_le90-62eb88b1.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90/roi_trans_r50_fpn_fp16_1x_dota_le90_20220303_193513.log.json)                     |
| ResNet50 (1024,1024,200)  | 76.08 | le90  |   1x    |   8.67   |      14.4      |   -   |     2      |                     [roi-trans-le90_r50_fpn_1x_dota](./roi-trans-le90_r50_fpn_1x_dota.py)                      |                               [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_le90/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_le90/roi_trans_r50_fpn_1x_dota_le90_20220130_132727.log.json)                               |
| Swin-tiny (1024,1024,200) | 77.51 | le90  |   1x    |          |      10.9      |   -   |     2      |               [roi-trans-le90_swin-tiny_fpn_1x_dota](./roi-trans-le90_swin-tiny_fpn_1x_dota.py)                |                   [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90/roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90/roi_trans_swin_tiny_fpn_1x_dota_le90_20220131_083622.log.json)                   |
| ResNet50 (1024,1024,500)  | 79.66 | le90  |   1x    |          |      14.4      | MS+RR |     2      |               [roi_trans_r50_fpn_1x_dota_ms_rr_le90](./roi_trans_r50_fpn_1x_dota_ms_rr_le90.py)                |                   [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_ms_rr_le90/roi_trans_r50_fpn_1x_dota_ms_rr_le90-fa99496f.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_ms_rr_le90/roi_trans_r50_fpn_1x_dota_ms_rr_le90_20220205_171729.log.json)                   |

Notes:

- `MS` means multiple scale image split.
- `RR` means random rotation.

## Citation

```
@InProceedings{ding2018learning,
	author = {Ding, Jian and Xue, Nan and Long, Yang and Xia, Gui-Song and Lu, Qikai},
	title = {Learning RoI Transformer for Oriented Object Detection in Aerial Images},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	pages={2849--2858},
	year = {2019}
}
```
