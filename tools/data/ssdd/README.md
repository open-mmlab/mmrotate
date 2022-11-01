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

The SSDD dataset can be downloaded from [Google drive](https://drive.google.com/file/d/1LmoHBk4xUvm0Zdtm8X7256dHigyFW4Nh/view?usp=sharing).

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
│   │   │   ├── all
│   │   │   ├── inshore
│   │   │   ├── offshore
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/ssdd.py` to `data/ssdd/`.
