# Preparing RSDD Dataset

<!-- [DATASET] -->

```bibtex
@ARTICLE{RSDD2022,
author = {C. Xu, H. Su, J. Li, Y. Liu, L. Yao, L. Gao, W. Yan and T. Wang},
title = {RSDD-SAR: Rotated Ship Detection Dataset in SAR Images},
journal = {Journal of Radars},
month = {Sep.},
year = {2022},
volume={11},
number={R22007},
pages={581},
}
```

## Download RSDD dataset

The RSDD dataset can be downloaded from [here:0518](https://pan.baidu.com/s/1_uezALB6eZ7DiPIozFoGJQ).

The data structure is as follows:

```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── rsdd
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   │   ├── JPEGValidation
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/rsdd.py` to `data/rsdd/`.
