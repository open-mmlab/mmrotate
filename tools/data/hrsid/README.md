# Preparing HRSID Dataset

<!-- [DATASET] -->

```bibtex
@ARTICLE{HRSID_2020,
author={Wei, Shunjun and Zeng, Xiangfeng and Qu, Qizhe and Wang, Mou and Su, Hao and Shi, Jun},
journal={IEEE Access},
title={HRSID: A High-Resolution SAR Images Dataset for Ship Detection and Instance Segmentation},
year={2020},
volume={8},
pages={120234-120254},
}
```

## Download HRSID dataset

The HRSID dataset can be downloaded from [Google drive](https://drive.google.com/file/d/1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE/view).

The data structure is as follows:

```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── HRSID_JPG
│   │   ├── JPEGImages
│   │   ├── annotations
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/hrisd.py` to `data/HRSID_JPG/`.
