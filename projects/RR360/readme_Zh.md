# RR360(360 Rotated Rectangle Detection)

本项目在[TRR360D](https://paperswithcode.com/dataset/trr360d)数据集上支持360度旋转矩形框目标检测

![image](https://user-images.githubusercontent.com/25839884/221288868-1bb8c3e5-818c-4228-b2dc-8dcfd33e7025.png)

# 环境配置

```shell
# 假设已经安装mmengine、mmcv 2.x、mmdetection
git clone https://github.com/open-mmlab/mmrotate -b dev-1.x
cd mmrotate
export MMROTATE_HOME=$(pwd)
pip install -v -e .
```

# 数据集准备

## 数据集下载

```shell
cd $MMROTATE_HOME
mkdir -p data/TRR360D
# git clone https://github.com/vansin/TRR360D.git data/TRR360D
wget https://openmmlab.vansin.top/datasets/ICDAR2019_MTD_HOQ.zip -O data/TRR360D/TRR360D.zip
cd $MMROTATE_HOME/data/TRR360D
unzip TRR360D.zip
```

600 个训练图片，240个测试图片

通过 browse_dataset.py 检测数据集的正确性

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/browse_dataset.py \
    projects/RR360/configs360/rotated_retinanet/rotated-retinanet-rbox-r360_r50_fpn_6x_ic19.py
```

数据集有由语义信息的，ABCD点是表格左上角开始，依次顺时钟的四个点。
![](https://cdn.vansin.top//picgo/3e4a042cd4b4725c4ae05aa7471467e.png)

# configs360

推理已训练好的模型

```shell
cd $MMROTATE_HOME

wget https://openmmlab.vansin.top/work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/epoch_36.pth

python projects/RR360/tools/test.py \
    projects/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    epoch_36.pth \
    --show-dir predict_result
    # --show

```

训练

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/train.py \
    projects/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    --d
```

测试自行训练的

```shell
cd $MMROTATE_HOME
python projects/RR360/tools/test.py \
    work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/rotated_rtmdet_s_l1-3x-ic19_pt.py \
    work_dirs/RR360/configs360/rotated_rtmdet_x3_r/rotated_rtmdet_s_l1-3x-ic19_pt/epoch_36.pth
```

# TODO

RotatedBoxes的r360的使用现在是通过以下不是很优雅的方式的实现的，未来将会使用注册器build的方式重构

```
from projects.RR360.structures.bbox import RotatedBoxes
import mmrotate.structures
# TODO : Refactoring with registry build
mmrotate.structures.bbox.RotatedBoxes = RotatedBoxes
```
