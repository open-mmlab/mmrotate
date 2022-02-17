# [ReDet: A Rotation-equivariant Detector for Aerial Object Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Han_ReDet_A_Rotation-Equivariant_Detector_for_Aerial_Object_Detection_CVPR_2021_paper.pdf)

<!-- [ALGORITHM] -->
## Abstract

![illustration](https://raw.githubusercontent.com/zytx121/image-host/main/imgs/redet.png)

Recently, object detection in aerial images has gained much attention in computer vision. Different from objects in natural images, aerial objects are often distributed with arbitrary orientation. Therefore, the detector requires more parameters to encode the orientation information, which are often highly redundant and inefficient. Moreover, as ordinary CNNs do not explicitly model the orientation variation, large amounts of rotation augmented data is needed to train an accurate object detector. In this paper, we propose a Rotation-equivariant Detector (ReDet) to address these issues, which explicitly encodes rotation equivariance and rotation invariance. More precisely, we incorporate rotation-equivariant networks into the detector to extract rotation-equivariant features, which can accurately predict the orientation and lead to a huge reduction of model size. Based on the rotation-equivariant features, we also present Rotation-invariant RoI Align (RiRoI Align), which adaptively extracts rotation-invariant features from equivariant features according to the orientation of RoI. Extensive experiments on several challenging aerial image datasets DOTA-v1.0, DOTA-v1.5 and HRSC2016, show that our method can achieve state-of-the-art performance on the task of aerial object detection. Compared with previous best results, our ReDet gains 1.2, 3.5 and 2.6 mAP on DOTA-v1.0, DOTA-v1.5 and HRSC2016 respectively while reducing the number of parameters by 60% (313 Mb vs. 121 Mb).




## Results and models

### DOTA1.0

|    Backbone   |    mAP   | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | Configs | Download |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------------:|
| ResNet50 (1024,1024,200) | 73.40 | le90 | 1x | 8.46 | 16.0 | - | 2 | [rotated_faster_rcnn_r50_fpn_1x_dota_le90](../rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90_20220131_082156.log.json)
| ReResNet50 (1024,1024,200) | 76.68| le90 | 1x | 9.32 | 4.0 | - | 2 | [redet_re50_refpn_1x_dota_le90](./redet_re50_refpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_le90/redet_re50_fpn_1x_dota_le90-724ab2da.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_le90/redet_re50_fpn_1x_dota_le90_20220130_132751.log.json)
| ReResNet50 (1024,1024,200) | 79.87 | le90 | 1x |  | 4.0 | MS+RR | 2 | [redet_re50_refpn_1x_dota_ms_rr_le90](./redet_re50_refpn_1x_dota_ms_rr_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_ms_rr_le90/redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_ms_rr_le90/redet_re50_fpn_1x_dota_ms_rr_le90_20220206_105343.log.json)

- `MS` means multiple scale image split.
- `RR` means random rotation.

Please download pretrained weight of ReResNet from [ReDet](https://github.com/csuhan/ReDet), and put it on `work_dirs/pretrain`. BTW, it is normal for `missing keys in source state_dict: xxx.filter ` to appear in the log. Don't worry!

## Citation
```
@inproceedings{han2021redet,
  title={Redet: A rotation-equivariant detector for aerial object detection},
  author={Han, Jiaming and Ding, Jian and Xue, Nan and Xia, Gui-Song},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2786--2795},
  year={2021}
}

```
