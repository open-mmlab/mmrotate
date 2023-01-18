# Preparing DIOR Dataset

<!-- [DATASET] -->

```bibtex
@article{LI2020296,
    title = {Object detection in optical remote sensing images: A survey and a new benchmark},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {159},
    pages = {296-307},
    year = {2020},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2019.11.023},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271619302825},
    author = {Ke Li and Gang Wan and Gong Cheng and Liqiu Meng and Junwei Han}
```

## Download DIOR dataset

The DIOR dataset can be downloaded from [here](https://gcheng-nwpu.github.io/#Datasets).

The data structure is as follows:

```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── DIOR
│   │   ├── JPEGImages-trainval
│   │   ├── JPEGImages-test
│   │   ├── Annotations
│   │   │   ├─ Oriented Bounding Boxes
│   │   │   ├─ Horizontal Bounding Boxes
│   │   ├── ImageSets
│   │   │   ├─ Main
│   │   │   │  ├─ train.txt
│   │   │   │  ├─ val.txt
│   │   │   │  ├─ test.txt
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/dior.py` to `data/dior/`.
