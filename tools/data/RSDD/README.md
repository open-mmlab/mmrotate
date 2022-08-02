# Preparing RSDD Dataset

<!-- [DATASET] -->

```bibtex
@articleInfo{R22007,
title = "RSDD-SAR: Rotated Ship Detection Dataset in SAR Images",
journal = "Journal of Radars",
volume = "11",
number = "R22007,
pages = "1",
year = "2022",
note = "",
issn = "2095-283X",
doi = "10.12000/JR22007",
url = "https://radars.ac.cn/en/article/doi/10.12000/JR22007",
author = "XU Congan","SU Hang","LI Jianwei","LIU Yu","YAO Libo","GAO Long","YAN Wenjun","WANG Taoyang",keywords = "Synthetic Aperture Radar (SAR)","Rotated SAR ship detection","Public dataset","RSDD-SAR","Deep learning",
```

## Download RSDD dataset

The RSDD dataset that have been converted to the format which can be used in mmrotate directly can be downloaded from [here:cire](https://pan.baidu.com/s/1vGr-xqMBGUTj9-8bNIQvwQ).

The data structure is as follows:

```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── rsdd
│   │   ├── train
│   │   ├── test
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/rsdd.py` to `data/rsdd/`.
