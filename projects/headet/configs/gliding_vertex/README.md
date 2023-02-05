# Gliding Vertex

> [Gliding Vertex on the Horizontal Bounding Box for Multi-Oriented Object Detection](https://arxiv.org/pdf/1911.09358.pdf)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/gv.png" width="800"/>
</div>

Object detection has recently experienced substantial progress. Yet, the widely adopted horizontal bounding box representation is not appropriate for ubiquitous oriented objects such as objects in aerial images and scene texts. In this paper, we propose a simple yet effective framework to detect multi-oriented objects. Instead of directly regressing the four vertices, we glide the vertex of the horizontal bounding box on each corresponding side to accurately describe a multi-oriented object. Specifically, We regress four length ratios characterizing the relative gliding offset on each corresponding side. This may facilitate the offset learning and avoid the confusion issue of sequential label points for oriented objects. To further remedy the confusion issue for nearly horizontal objects, we also introduce an obliquity factor based on area ratio between the object and its horizontal bounding box, guiding the selection of horizontal or oriented detection for each object. We add these five extra target variables to the regression head of rotated faster R-CNN, which requires ignorable extra computation time. Extensive experimental results demonstrate that without bells and whistles, the proposed method achieves superior performances on multiple multi-oriented object detection benchmarks including object detection in aerial images, scene text detection, pedestrian detection in fisheye images.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 73.40 | le90  |   1x    |   8.46   |      16.5      |  -  |     2      | [rotated-faster-rcnn-le90_r50_fpn_1x_dota](../rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90_20220131_082156.log.json) |
| ResNet50 (1024,1024,200) | 73.23 | le90  |   1x    |   8.45   |      16.4      |  -  |     2      |                [gliding-vertex-rbox_r50_fpn_1x_dota](./gliding-vertex-rbox_r50_fpn_1x_dota.py)                 |                [model](https://download.openmmlab.com/mmrotate/v0.1.0/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90/gliding_vertex_r50_fpn_1x_dota_le90-12e7423c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90/gliding_vertex_r50_fpn_1x_dota_le90_20220129_085529.log.json)                |

## Citation

```
@article{RN57,
	author = {Xu, Yongchao and Fu, Mingtao and Wang, Qimeng and Wang, Yukang and Chen, Kai and Xia, Gui-Song and Bai, Xiang},
	title = {Gliding Vertex on the Horizontal Bounding Box for Multi-Oriented Object Detection},
	journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
	volume = {43},
	number = {4},
	pages = {1452-1459},
	ISSN = {0162-8828},
	DOI = {10.1109/tpami.2020.2974745},
	year = {2021},
	type = {Journal Article}
}
```
