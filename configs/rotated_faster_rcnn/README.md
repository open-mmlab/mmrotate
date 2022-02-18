# [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)


<!-- [ALGORITHM] -->
## Abstract

![illustration](https://raw.githubusercontent.com/zytx121/image-host/main/imgs/faster_rcnn.png)

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet [7] and Fast R-CNN [5] have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully-convolutional network that simultaneously predicts object bounds and objectness scores at each position. RPNs are trained end-to-end to generate highquality region proposals, which are used by Fast R-CNN for detection. With a simple alternating optimization, RPN and Fast R-CNN can be trained to share convolutional features. For the very deep VGG-16 model [19], our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007 (73.2% mAP) and 2012 (70.4% mAP) using 300 proposals per image.


## Results and models

### DOTA1.0

|    Backbone   |    mAP   | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | Configs | Download |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------------:|
| ResNet50 (1024,1024,200) | 73.40 | le90 | 1x | 8.46 | 16.0 | - | 2 | [rotated_faster_rcnn_r50_fpn_1x_dota_le90](../rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth) &#124; [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90_20220131_082156.log.json)

## Citation
```
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```
