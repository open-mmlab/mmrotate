# Rotated YOLOX

> [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

<!-- [ALGORITHM] -->

## Abstract

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector --
YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a
decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale
range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet
by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO,
outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as
YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP.
Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a
single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical
scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001736-9fb303dd-eac7-46b0-ad45-214cfa51e928.png"/>
</div>

## Results and Models

|    Backbone     |    Bbox Loss Type    |    Size     |  mAP  | FPS  |                      Config                       | Download |
| :-------------: | :------------------: | :---------: | :---: | :--: | :-----------------------------------------------: | :------: |
| Rotated YOLOX-s |     Rotated IoU      | (1024,1024) | 74.36 | 53.1 |   [config](./rotated_yolox_s_300e_dota_le90.py)   |    -     |
| Rotated YOLOX-s | Horizontal IoU + CSL | (1024,1024) |   -   |  -   | [config](./rotated_yolox_s_csl_300e_dota_le90.py) |    -     |
| Rotated YOLOX-s |         KLD          | (1024,1024) |   -   |  -   | [config](./rotated_yolox_s_kld_300e_dota_le90.py) |    -     |

**Note**:

- Compared with original YOLOX in mmdet, Rotated YOLOX enable `grad_clip` to prevent nan at training process.
- All models are trained with batch size 8 on one GPU.
- FPS and speed are tested on a single RTX3090.

## Citation

```latex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
