# Oriented RepPoints

> [Oriented RepPoints for Aerial Object Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Oriented_RepPoints_for_Aerial_Object_Detection_CVPR_2022_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/oriented_reppoints.png" width="800"/>
</div>

In contrast to the generic object, aerial targets are often non-axis aligned with arbitrary orientations having
the cluttered surroundings. Unlike the mainstreamed approaches regressing the bounding box orientations, this paper
proposes an effective adaptive points learning approach to aerial object detection by taking advantage of the adaptive
points representation, which is able to capture the geometric information of the arbitrary-oriented instances.
To this end, three oriented conversion functions are presented to facilitate the classification and localization
with accurate orientation. Moreover, we propose an effective quality assessment and sample assignment scheme for
adaptive points learning toward choosing the representative oriented reppoints samples during training, which is
able to capture the non-axis aligned features from adjacent objects or background noises. A spatial constraint is
introduced to penalize the outlier points for roust adaptive learning. Experimental results on four challenging
aerial datasets including DOTA, HRSC2016, UCAS-AOD and DIOR-R, demonstrate the efficacy of our proposed approach.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                  Configs                                                  |                                                                                                                                                                                    Download                                                                                                                                                                                    |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 71.94 | le135 |   1x    |   3.45   |      16.1      |  -  |     2      |          [oriented-reppoints-qbox_r50_fpn_1x_dota](./oriented-reppoints-qbox_r50_fpn_1x_dota.py)          |         [model](https://download.openmmlab.com/mmrotate/v0.1.0/orientedreppoints/oriented_reppoints_r50_fpn_1x_dota_le135/oriented_reppoints_r50_fpn_1x_dota_le135-ef072de9.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/orientedreppoints/oriented_reppoints_r50_fpn_1x_dota_le135/oriented_reppoints_r50_fpn_1x_dota_le135_20220505_092853.log.json)         |
| ResNet50 (1024,1024,200) | 75.21 | le135 |   40e   |   3.45   |      16.1      | ms  |     2      | [oriented-reppoints-qbox_r50_fpn_mstrain-40e_dota](./oriented-reppoints-qbox_r50_fpn_mstrain-40e_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/orientedreppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/orientedreppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135/oriented_reppoints_r50_fpn_40e_dota_ms_le135_20220516_145332.log.json) |

**Notes:**

- Oriented RepPoints needs to install [MMCV](https://github.com/open-mmlab/mmcv) >= 1.5.3.
- `ms` means multi-scale image split online (768, 1280).

## Citation

```
@inproceedings{li2022ori,
    title={Oriented RepPoints for Aerial Object Detection},
    author={Wentong Li, Yijie Chen, Kaixuan Hu, Jianke Zhu},
    booktitle={Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```
