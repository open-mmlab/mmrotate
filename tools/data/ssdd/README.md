# Preparing SSDD Dataset

<!-- [DATASET] -->

```bibtex
@ARTICLE{SSDD2021,
author = {T. Zhang, X. Zhang, J. Li and X. Xu},
title = {SAR ship detection dataset (SSDD): Official release and comprehensive data analysis},
journal = {Remote Senseing},
month = {Sep.},
year = {2021}
volume={13},
number={18},
pages={3690},
}
```

## Download SSDD dataset
The SSDD dataset can be downloaded from [here:0518](https://pan.baidu.com/s/1_uezALB6eZ7DiPIozFoGJQ).

The data structure is as follows:
```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── ssdd
│   │   ├── train
│   │   ├── test
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/ssdd.py` to `data/ssdd/`.
