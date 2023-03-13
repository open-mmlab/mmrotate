# RR360(360 Rotated Rectangle Detection)

This project supports 360-degree rotated rectangle detection on the [TRR360D](https://paperswithcode.com/dataset/trr360d) dataset.

![image](https://user-images.githubusercontent.com/25839884/221288868-1bb8c3e5-818c-4228-b2dc-8dcfd33e7025.png)

# Environment Setup

```shell
# Assuming mmengine, mmcv 2.x, and mmdetection are already installed
git clone https://github.com/open-mmlab/mmrotate -b dev-1.x
cd mmrotate
export MMROTATE_HOME=$(pwd)
pip install -v -e .
```

# Dataset Preparation

## Dataset Download

```shell
cd $MMROTATE_HOME
mkdir -p data/TRR360D
# git clone https://github.com/vansin/TRR360D.git data/TRR360D
wget https://openmmlab.vansin.top/datasets/ICDAR2019_MTD_HOQ.zip -O data/TRR360D/TRR360D.zip
cd $MMROTATE_HOME/data/TRR360D
unzip TRR360D.zip
```

600 training images and 240 test images

Check the correctness of the dataset through browse_dataset.py

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/browse_dataset.py \
    projects/RR360/configs360/rotated_retinanet/rotated-retinanet-rbox-r360_r50_fpn_6x_ic19.py
```

The dataset has semantic information, and points ABCD are the top-left corner of the table, followed by the other three points clockwise.
![](https://cdn.vansin.top//picgo/3e4a042cd4b4725c4ae05aa7471467e.png)

# configs360

Infer the trained model

```shell
cd $MMROTATE_HOME

wget https://openmmlab.vansin.top/work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/epoch_36.pth

python projects/RR360/tools/test.py \
    projects/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    epoch_36.pth \
    --show-dir predict_result
    # --show

```

Train the model

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/train.py \
    projects/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    --d
```

Test the self-trained model

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/test.py \
    work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/epoch_36.pth
```

# TODO

The usage of RotatedBoxes with r360 is currently implemented through the following less elegant approach, but it will be refactored in the future using the registry build method.

```
from projects.RR360.structures.bbox import RotatedBoxes
import mmrotate.structures
# TODO : Refactoring with registry build
mmrotate.structures.bbox.RotatedBoxes = RotatedBoxes
```
