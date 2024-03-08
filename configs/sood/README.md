# SOOD

> [SOOD: Towards Semi-Supervised Oriented Object Detection](https://arxiv.org/abs/2304.04515)

<!-- [ALGORITHM] -->

## Abstract

Semi-Supervised Object Detection (SSOD), aiming to explore unlabeled data for boosting object detectors, has become an active task in recent years. However, existing SSOD approaches mainly focus on horizontal objects, leaving multi-oriented objects that are common in aerial images unexplored. This paper proposes a novel Semi-supervised Oriented Object Detection model, termed SOOD, built upon the mainstream pseudo-labeling framework. Towards oriented objects in aerial scenes, we design two loss functions to provide better supervision. Focusing on the orientations of objects, the first loss regularizes the consistency between each pseudo-label-prediction pair (includes a prediction and its corresponding pseudo label) with adaptive weights based on their orientation gap. Focusing on the layout of an image, the second loss regularizes the similarity and explicitly builds the many-to-many relation between the sets of pseudo-labels and predictions. Such a global consistency constraint can further boost semi-supervised learning. Our experiments show that when trained with the two proposed losses, SOOD surpasses the state-of-the-art SSOD methods under various settings on the DOTA-v1.5 benchmark.

## Requirements

- `mmpretrain>=1.0.0`
  please refer to [mmpretrain](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for installation.

## Data Preparation

Please refer to [data_preparation.md](tools/data/dota/README.md) to prepare the original data. After that, the data folder should be organized as follows:

```
├── data
│   ├── split_ss_dota1_5
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```

For partial labeled setting, we split the DOTA-v1.5's train set via the author released [split data list](tools/misc/split_dota1.5_lists) and [split tool](tools/misc/split_dota1.5_via_lists.py)

```angular2html
python tools/misc/split_dota1.5_via_lists.py
```

For fully labeled setting, we use DOTA-V1.5 train as labeled set and DOTA-V1.5 test as unlabeled set.

After that, the data folder should be organized as follows:

```
├── data
│   ├── split_ss_dota1_5
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_10_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_10_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_20_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_20_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_30_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_30_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```

## Results

DOTA1.5

|         Backbone         | Setting | mAP50 | Mem (GB) |                             Config                              |
| :----------------------: | :-----: | :---: | :------: | :-------------------------------------------------------------: |
| ResNet50 (1024,1024,200) |   10%   |       |   8.45   | [config](./sood_fcos_r50_fpn_2xb3-180000k_semi-0.1-dotav1.5.py) |
| ResNet50 (1024,1024,200) |   20%   |       |          | [config](./sood_fcos_r50_fpn_2xb3-180000k_semi-0.2-dotav1.5.py) |
| ResNet50 (1024,1024,200) |   30%   |       |          | [config](./sood_fcos_r50_fpn_2xb3-180000k_semi-0.3-dotav1.5.py) |

## Citation

```
@inproceedings{hua2023sood,
  title={SOOD: Towards Semi-Supervised Oriented Object Detection},
  author={Hua, Wei and Liang, Dingkang and Li, Jingyu and Liu, Xiaolong and Zou, Zhikang and Ye, Xiaoqing and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15558--15567},
  year={2023}
}
```
