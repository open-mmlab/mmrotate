# [Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence](https://arxiv.org/pdf/2106.01883.pdf)

<!-- [ALGORITHM] -->
## Abstract

![illustration](https://raw.githubusercontent.com/zytx121/image-host/main/imgs/kld.png)

Existing rotated object detectors are mostly inherited from the horizontal detection paradigm, as the latter has evolved into a well-developed area. However, these detectors are difficult to perform prominently in high-precision detection due to the limitation of current regression loss design, especially for objects with large aspect ratios. Taking the perspective that horizontal detection is a special case for rotated object detection, in this paper, we are motivated to change the design of rotation regression loss from induction paradigm to deduction methodology, in terms of the relation between rotation and horizontal detection. We show that one essential challenge is how to modulate the coupled parameters in the rotation regression loss, as such the estimated parameters can influence to each other during the dynamic joint optimization, in an adaptive and synergetic way. Specifically, we first convert the rotated bounding box into a 2-D Gaussian distribution, and then calculate the Kullback-Leibler Divergence (KLD) between the Gaussian distributions as the regression loss. By analyzing the gradient of each parameter, we show that KLD (and its derivatives) can dynamically adjust the parameter gradients according to the characteristics of the object. For instance, it will adjust the importance (gradient weight) of the angle parameter according to the aspect ratio. This mechanism can be vital for high-precision detection as a slight angle error would cause a serious accuracy drop for large aspect ratios objects. More importantly, we have proved that KLD is scale invariant. We further show that the KLD loss can be degenerated into the popular $l_{n}$-norm loss for horizontal detection. Experimental results on seven datasets using different detectors show its consistent superiority

## Results and models

### DOTA1.0

#### RotatedRetinaNet
|    Backbone   |    mAP   | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | Configs | Download |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------------:|
| ResNet50 (1024,1024,200) | 64.55 | oc | 1x | 3.38 | 14.8 | - | 2 | [rotated_retinanet_hbb_r50_fpn_1x_dota_oc](../rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py) |  [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc_20220121_095315.log.json)
| ResNet50 (1024,1024,200) | 69.94 | oc | 1x | 3.39 | 14.9 | - | 2 | [rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc](./rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc.py) |  [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc-49c1f937.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc_20220125_201832.log.json)

|    Backbone   |    mAP   | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | Configs | Download |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------------:|
| ResNet50 (1024,1024,200) | 69.80 | oc | 1x | 3.54 | 12.1 | - | 2 | [r3det_r50_fpn_1x_dota_oc](../r3det/r3det_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc-b1fb045c.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc_20220126_191226.log.json)
| ResNet50 (1024,1024,200) | 71.83 | oc | 1x | 3.54 | 12.2 | - | 2 | [r3det_kld_r50_fpn_1x_dota_oc](./r3det_kld_r50_fpn_1x_dota_oc.py) |  [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_kld_r50_fpn_1x_dota_oc/r3det_kld_r50_fpn_1x_dota_oc-31866226.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_kld_r50_fpn_1x_dota_oc/r3det_kld_r50_fpn_1x_dota_oc_20220210_114049.log.json)

#### R3Det*
|    Backbone   |    mAP   | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | Configs | Download |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------------:|
| ResNet50 (1024,1024,200) | 70.18 | oc | 1x | 3.23 | 15.1 | - | 2 | [r3det_tiny_r50_fpn_1x_dota_oc](../r3det/r3det_tiny_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_tiny_r50_fpn_1x_dota_oc/r3det_tiny_r50_fpn_1x_dota_oc-c98a616c.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_tiny_r50_fpn_1x_dota_oc/r3det_tiny_r50_fpn_1x_dota_oc_20220209_171624.log.json)
| ResNet50 (1024,1024,200) | 72.76 | oc | 1x | 3.44 | 13.6 | - | 2 | [r3det_tiny_kld_r50_fpn_1x_dota_oc](./r3det_tiny_kld_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_tiny_kld_r50_fpn_1x_dota_oc/r3det_tiny_kld_r50_fpn_1x_dota_oc-589e142a.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_tiny_kld_r50_fpn_1x_dota_oc/r3det_tiny_kld_r50_fpn_1x_dota_oc_20220209_172917.log.json)

## Citation
```
@inproceedings{yang2021learning,
	title={Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence},
	author={Yang, Xue and Yang, Xiaojiang and Yang, Jirui and Ming, Qi and Wang, Wentao and Tian, Qi and Yan, Junchi},
	booktitle={Advances in Neural Information Processing Systems},
	year={2021}
}
```
