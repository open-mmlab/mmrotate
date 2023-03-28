# LSKNet

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-selective-kernel-network-for-remote/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=large-selective-kernel-network-for-remote)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-selective-kernel-network-for-remote/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=large-selective-kernel-network-for-remote)

## Abstract

Recent research on remote sensing object detection has largely focused on improving the representation of oriented bounding boxes but has overlooked the unique prior knowledge presented in remote sensing scenarios. Such prior knowledge can be useful because tiny remote sensing objects may be mistakenly detected without referencing a sufficiently long-range context, and the long-range context required by different types of objects can vary. In this paper, we take these priors into account and propose the Large Selective Kernel Network (LSKNet). LSKNet can dynamically adjust its large spatial receptive field to better model the ranging context of various objects in remote sensing scenarios. To the best of our knowledge, this is the first time that large and selective kernel mechanisms have been explored in the field of remote sensing object detection. Without bells and whistles, LSKNet sets new state-of-the-art scores on standard benchmarks, i.e., HRSC2016 (98.46% mAP), DOTA-v1.0 (81.85% mAP) and FAIR1M-v1.0 (47.87% mAP). Based on a similar technique, we rank 2nd place in 2022 the Greater Bay Area International Algorithm Competition

## Description

Author: @Yuxuan Li.
This project is an implementation of "Large Selective Kernel Network for Remote Sensing Object Detection" at: [https://arxiv.org/pdf/2303.09030.pdf](https://arxiv.org/pdf/2303.09030.pdf)

## Usage

### Training commands

In MMRotate's root directory, run the following command to train the model:

```bash
python tools/train.py projects/LSKNet/configs/lsk_t_fpn_1x_dota_le90.py
```

### Testing commands

In MMRotate's root directory, run the following command to test the model:

```bash
python tools/test.py projects/LSKNet/configs/lsk_t_fpn_1x_dota_le90.py ${CHECKPOINT_PATH}
```

## Results

Imagenet 300-epoch pre-trained LSKNet-T backbone: [Download](https://download.openmmlab.com/mmrotate/v1.0/lsknet/backbones/lsk_t_backbone-2ef8a593.pth)

Imagenet 300-epoch pre-trained LSKNet-S backbone: [Download](https://download.openmmlab.com/mmrotate/v1.0/lsknet/backbones/lsk_s_backbone-e9d2e551.pth)

DOTA1.0

|                           Model                            |  mAP  | Angle | lr schd | Batch Size |                                   Configs                                    |                                                                                                                                 Download                                                                                                                                  |     note     |
| :--------------------------------------------------------: | :---: | :---: | :-----: | :--------: | :--------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------: |
| [RTMDet-l](https://arxiv.org/abs/2212.07784) (1024,1024,-) | 81.33 |   -   | 3x-ema  |     8      |                                      -                                       |                                                                                                                                     -                                                                                                                                     |  Prev. Best  |
|                  LSKNet_T (1024,1024,200)                  | 81.37 | le90  |   1x    |    2\*8    |     [lsk_t_fpn_1x_dota_le90](./configs/lsknet/lsk_t_fpn_1x_dota_le90.py)     |         [model](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_t_fpn_1x_dota_le90/lsk_t_fpn_1x_dota_le90_20230206-3ccee254.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_t_fpn_1x_dota_le90/lsk_t_fpn_1x_dota_le90_20230206.log)         |              |
|                  LSKNet_S (1024,1024,200)                  | 81.64 | le90  |   1x    |    1\*8    |     [lsk_s_fpn_1x_dota_le90](./configs/lsknet/lsk_s_fpn_1x_dota_le90.py)     |         [model](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_fpn_1x_dota_le90/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_fpn_1x_dota_le90/lsk_s_fpn_1x_dota_le90_20230116.log)         |              |
|                 LSKNet_S\* (1024,1024,200)                 | 81.85 | le90  |   1x    |    1\*8    | [lsk_s_ema_fpn_1x_dota_le90](./configs/lsknet/lsk_s_ema_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_ema_fpn_1x_dota_le90/lsk_s_ema_fpn_1x_dota_le90_20230212-30ed4041.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_ema_fpn_1x_dota_le90/lsk_s_ema_fpn_1x_dota_le90_20230212.log) | EMA Finetune |

<!-- https://github.com/open-mmlab/mmdetection/tree/3.x/configs/rtmdet -->

HRSC2016

|                    Model                     | mAP(07) | mAP(12) | Angle | lr schd | Batch Size |                                      Configs                                      |                                                                                                                              Download                                                                                                                              |    note    |
| :------------------------------------------: | :-----: | :-----: | :---: | :-----: | :--------: | :-------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: |
| [RTMDet-l](https://arxiv.org/abs/2212.07784) |  90.60  |  97.10  | le90  |   3x    |     -      |                                         -                                         |                                                                                                                                 -                                                                                                                                  | Prev. Best |
|  [ReDet](https://arxiv.org/abs/2103.07733)   |  90.46  |  97.63  | le90  |   3x    |    2\*4    | [redet_re50_refpn_3x_hrsc_le90](./configs/redet/redet_re50_refpn_3x_hrsc_le90.py) |                                                                                                                                 -                                                                                                                                  | Prev. Best |
|                   LSKNet_S                   |  90.65  |  98.46  | le90  |   3x    |    1\*8    |       [lsk_s_fpn_3x_hrsc_le90](./configs/lsknet/lsk_s_fpn_3x_hrsc_le90.py)        | [model](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_fpn_3x_hrsc_le90/lsk_s_fpn_3x_hrsc_le90_20230205-4a4a39ce.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_fpn_3x_hrsc_le90/lsk_s_fpn_3x_hrsc_le90_20230205-4a4a39ce.pth) |            |

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{li2023large,
  title   = {Large Selective Kernel Network for Remote Sensing Object Detection},
  author  = {Li, Yuxuan and Hou, Qibin and Zheng, Zhaohui and Cheng, Mingming and Yang, Jian and Li, Xiang},
  journal={ArXiv},
  year={2023}
}
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmrotate.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmrotate/blob/766185ed317f99379cb14035a6f9e5cf8a5340ad/mmrotate/structures/bbox/box_converters.py#L61-L78) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmrotate/blob/766185ed317f99379cb14035a6f9e5cf8a5340ad/tests/test_structures/test_bbox/test_box_converters.py#L43-L52) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/r3det/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/r3det/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
