# CFA

> [Beyond Bounding-Box: Convex-hull Feature Adaptation for Oriented and Densely Packed Object Detection.](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Beyond_Bounding-Box_Convex-Hull_Feature_Adaptation_for_Oriented_and_Densely_Packed_CVPR_2021_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/cfa.png" width="800"/>
</div>

Detecting oriented and densely packed objects remains challenging for spatial feature aliasing caused by the intersection of reception fields between objects. In this paper, we propose a convex-hull feature adaptation (CFA) approach for configuring convolutional features in accordance with oriented and densely packed object layouts. CFA is rooted in convex-hull feature representation, which defines a set of dynamically predicted feature points guided by the convex intersection over union (CIoU) to bound the extent of objects. CFA pursues optimal feature assignment by constructing convex-hull sets and dynamically splitting positive or negative convex-hulls. By simultaneously considering overlapping convex-hulls and objects and penalizing convex-hulls shared by multiple objects, CFA alleviates spatial feature aliasing towards optimal feature adaptation. Experiments on DOTA and SKU110KR datasets show that CFA significantly outperforms the baseline approach, achieving new state-of-the-art detection performance.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                 Configs                                                  |                                                                                                                                                                    Download                                                                                                                                                                    |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 59.44 |  oc   |   1x    |   3.45   |      15.6      |  -  |     2      | [rotated-reppoints-qbox_r50_fpn_1x_dota](../rotated_reppoints/rotated-reppoints-qbox_r50_fpn_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc-d38ce217.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc_20220205_145010.log.json) |
| ResNet50 (1024,1024,200) | 69.63 | le135 |   1x    |   3.45   |      16.1      |  -  |     2      |                        [cfa-qbox_r50_fpn_1x_dota](./cfa-qbox_r50_fpn_1x_dota.py)                         |                                     [model](https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_1x_dota_le135/cfa_r50_fpn_1x_dota_le135-aed1cbc6.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_1x_dota_le135/cfa_r50_fpn_1x_dota_le135_20220205_144859.log.json)                                     |
| ResNet50 (1024,1024,200) | 73.45 |  oc   |   40e   |   3.45   |      16.1      |  -  |     2      |                       [cfa-qbox_r50_fpn_40e_dota](./cfa-qbox_r50_fpn_40e_dota.py)                        |                                         [model](https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_40e_dota_oc/cfa_r50_fpn_40e_dota_oc-2f387232.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_40e_dota_oc/cfa_r50_fpn_40e_dota_oc_20220209_171237.log.json)                                         |

## Citation

```
@inproceedings{Guo_2021CVPR_CFA,
  author    = {Zonghao Guo, Chang Liu, Xiaosong Zhang, Jianbin Jiao, Xiangyang Ji and Qixiang Ye},
  title     = {Beyond Bounding-Box: Convex-hull Feature Adaptation for Oriented and Densely Packed Object Detection},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2021}
}
```
