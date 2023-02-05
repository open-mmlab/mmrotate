# SASM

> [Shape-Adaptive Selection and Measurement for Oriented Object Detection](https://www.aaai.org/AAAI22Papers/AAAI-2171.HouL.pdf)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/sasm.jpg" width="800"/>
</div>

The development of detection methods for oriented object detection remains a challenging task. A considerable obstacle
is the wide variation in the shape (e.g., aspect ratio) of objects. Sample selection in general object detection has been
widely studied as it plays a crucial role in the performance of the detection method and has achieved great progress.
However, existing sample selection strategies still overlook some issues: (1) most of them ignore the object shape information;
(2) they do not make a potential distinction between selected positive samples; and (3) some of them can only be applied
to either anchor-free or anchor-based methods and cannot be used for both of them simultaneously. In this paper, we
propose novel flexible shape-adaptive selection (SA-S) and shape-adaptive measurement (SA-M) strategies for oriented
object detection, which comprise an SA-S strategy for sample selection and SA-M strategy for the quality estimation of
positive samples. Specifically, the SA-S strategy dynamically selects samples according to the shape information and
characteristics distribution of objects. The SA-M strategy measures the localization potential and adds quality information
on the selected positive samples. The experimental results on both anchor-free and anchor-based baselines and four publicly
available oriented datasets (DOTA, HRSC2016, UCASAOD, and ICDAR2015) demonstrate the effectiveness of the proposed method

## Results and models

DOTA1.0

#### RepPoints

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                 Configs                                                  |                                                                                                                                                                    Download                                                                                                                                                                    |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 59.44 |  oc   |   1x    |   3.45   |      15.6      |  -  |     2      | [rotated-reppoints-qbox_r50_fpn_1x_dota](../rotated_reppoints/rotated-reppoints-qbox_r50_fpn_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc-d38ce217.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc_20220205_145010.log.json) |
| ResNet50 (1024,1024,200) | 66.45 |  oc   |   1x    |   3.53   |      15.3      |  -  |     2      |             [sasm-reppoints-qbox_r50_fpn_1x_dota](./sasm-reppoints-qbox_r50_fpn_1x_dota.py)              |                    [model](https://download.openmmlab.com/mmrotate/v0.1.0/sasm/sasm_reppoints_r50_fpn_1x_dota_oc/sasm_reppoints_r50_fpn_1x_dota_oc-6d9edded.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/sasm/sasm_reppoints_r50_fpn_1x_dota_oc/sasm_reppoints_r50_fpn_1x_dota_oc_20220205_144938.log.json)                    |

## Citation

```
@inproceedings{hou2022shape,
    title={Shape-Adaptive Selection and Measurement for Oriented Object Detection},
    author={Hou, Liping and Lu, Ke and Xue, Jian and Li, Yuqiu},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2022}
}

```
