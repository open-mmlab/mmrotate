# Rotated ATSS

> [Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection](https://arxiv.org/abs/1912.02424)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/atss.jpg" width="800"/>
</div>

Object detection has been dominated by anchor-based detectors for several years. Recently, anchor-free detectors have become popular due to the proposal of FPN and Focal Loss. In this paper, we first point out that the essential difference between anchor-based and anchor-free detection is actually how to define positive and negative training samples, which leads to the performance gap between them. If they adopt the same definition of positive and negative samples during training, there is no obvious difference in the final performance, no matter regressing from a box or a point. This shows that how to select positive and negative training samples is important for current object detectors. Then, we propose an Adaptive Training Sample Selection (ATSS) to automatically select positive and negative samples according to statistical characteristics of object. It significantly improves the performance of anchor-based and anchor-free detectors and bridges the gap between them. Finally, we discuss the necessity of tiling multiple anchors per location on the image to detect objects. Extensive experiments conducted on MS COCO support our aforementioned analysis and conclusions. With the newly introduced ATSS, we improve state-of-the-art detectors by a large margin to 50.7% AP without introducing any overhead.

## Results and Models

DOTA1.0

Notes:

- `hbb` means the input of the assigner is the predicted box and the horizontal box that can surround the GT. `obb` means the input of the assigner is the predicted box and the GT. They can be switched by `assign_by_circumhbbox`  in `RotatedRetinaHead`.

## Citation

```
@inproceedings{zhang2020bridging,
  title={Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection},
  author={Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9759--9768},
  year={2020}
}
```
