# ConvNeXt

> [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

## Abstract

The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/148624004-e9581042-ea4d-4e10-b3bd-42c92b02053b.png" width="90%"/>
</div>

## Results and models

DOTA1.0

|          Backbone          |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                                        Configs                                                                        |                                                                                                                                                                                                                   Download                                                                                                                                                                                                                   |
| :------------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  ResNet50 (1024,1024,200)  | 70.22 | le90  |   1x    |   3.35   |      16.9      |  -  |     2      |              [rotated-retinanet-rbox-le90_r50_fpn_kld-stable_1x_dota](../kld/rotated-retinanet-rbox-le90_r50_fpn_kld-stable_1x_dota.py)               |                            [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90-31193e00.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90_20220402_225531.log.json)                            |
|  ResNet50 (1024,1024,200)  | 71.30 | le90  |   1x    |   3.61   |      16.9      |  -  |     2      |        [rotated-retinanet-rbox-le90_r50_fpn_kld-stable_adamw-1x_dota](../kld/rotated-retinanet-rbox-le90_r50_fpn_kld-stable_adamw-1x_dota.py)         |                [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90-474d9955.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90_20220608_003758.log.json)                |
| ConvNeXt-T (1024,1024,200) | 74.49 | le90  |   1x    |   6.12   |      7.9       |  -  |     2      | [rotated-retinanet-rbox-le90_convnext-tiny_fpn_kld-stable_adamw-1x_dota](./rotated-retinanet-rbox-le90_convnext-tiny_fpn_kld-stable_adamw-1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/convnext/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90-388184f6.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/convnext/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90_20220608_191712.log.json) |

**Note**:

- ConvNeXt backbone needs to install [MMClassification](https://github.com/open-mmlab/mmclassification) first, which has abundant backbones for downstream tasks.

```shell
pip install "mmcls>=1.0.0rc0"
```

- The performance may be unstable according to mmdetection's experience.

## Citation

```bibtex
@article{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
