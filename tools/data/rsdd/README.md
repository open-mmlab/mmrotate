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

The RSDD dataset can be downloaded from [Google drive](https://drive.google.com/file/d/1PJxr7Tbr_ZAzuG8MNloDa4mLaRYCD3qc/view?usp=sharing).

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
