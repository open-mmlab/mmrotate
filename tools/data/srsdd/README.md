# Preparing SRSDD Dataset

<!-- [DATASET] -->

```bibtex
@ARTICLE{SRSDD2021,
author = {S. Lei, D. Lu and X. Qiu},
title = {SRSDD-v1.0: A high-resolution SAR rotation ship
detection dataset},
journal = {Remote Senseing},
month = {Dec.},
year = {2021},
volume={13},
number={24},
pages={5104},
}
```

## Download SRSDD dataset

The SRSDD dataset can be downloaded from [Google drive](https://drive.google.com/file/d/1QtCjih1ChOmG-TOPUTlsL3WbMh0L-1zp/view?usp=sharing).

The data structure is as follows:

```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── srsdd
│   │   ├── train
│   │   ├── test
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/srsdd.py` to `data/srsdd/`.
