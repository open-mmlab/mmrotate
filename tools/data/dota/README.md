# Preparing DOTA Dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{Xia_2018_CVPR,
author = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
title = {DOTA: A Large-Scale Dataset for Object Detection in Aerial Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

## download dota dataset
The dota dataset can be downloaded from [here](https://captain-whu.github.io/DOTA/dataset.html).

The data structure is as follows:
```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── DOTA
│   │   ├── train
│   │   ├── val
│   │   ├── test
```

## split dota dataset

Please crop the original images into 1024×1024 patches with an overlap of 200 by run

```shell
python tools/data/dota/split/img_split.py --base_json \
  tools/data/dota/split/split_configs/dota1_0/ss_trainval.json

python tools/data/dota/split/img_split.py --base_json \
  tools/data/dota/split/split_configs/dota1_0/ss_test.json
```

If you want to get a multiple scale dataset, you can run the following command.
```shell
python tools/data/dota/split/img_split.py --base_json \
  tools/data/dota/split/split_configs/dota1_0/ms_trainval.json

python tools/data/dota/split/img_split.py --base_json \
  tools/data/dota/split/split_configs/dota1_0/ms_test.json
```

Please change the `img_dirs` and `ann_dirs` in json. (Forked from [BboxToolkit](https://github.com/jbwang1997/BboxToolkit), which is faster then DOTA_Devkit.)


## change root path in base config

Please change `data_root` in `configs/_base_/datasets/dota1_0.py` to split DOTA dataset.
