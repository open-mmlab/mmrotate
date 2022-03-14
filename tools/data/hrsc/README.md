# Preparing HRSC Dataset

<!-- [DATASET] -->

```bibtex
@conference{hrsc,
    author = {Zikun Liu. and Liu Yuan. and Lubin Weng. and Yiping Yang.},
    title = {A High Resolution Optical Satellite Image Dataset for Ship Recognition and Some New Baselines},
    booktitle = {Proceedings of the 6th International Conference on Pattern Recognition Applications and Methods - ICPRAM,},
    year = {2017},
    pages = {324-331},
    publisher = {SciTePress},
    organization = {INSTICC},
    doi = {10.5220/0006120603240331},
    isbn = {978-989-758-222-6},
    issn = {2184-4313},
}
```

## Download HRSC dataset

The HRSC dataset can be downloaded from [here](https://aistudio.baidu.com/aistudio/datasetdetail/54106).

The data structure is as follows:

```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── hrsc
│   │   ├── FullDataSet
│   │   │   ├─ AllImages
│   │   │   ├─ Annotations
│   │   │   ├─ LandMask
│   │   │   ├─ Segmentations
│   │   ├── ImageSets
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/hrsc.py` to `data/hrsc/`.
