<div align="center">
  <img src="resources/mmrotate-logo.png" width="500px"/>
</div>

## Introduction

English | [简体中文](README_zh-CN.md)

# MMRotate: OpenMMlab Rotating Object Detection Toolkit

[![Documentation](https://readthedocs.org/projects/mmrotate/badge/?version=latest)](https://mmrotate.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmrotate/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrotate/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrotate/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrotate)
[![PyPI](https://badge.fury.io/py/mmrotate.svg)](https://pypi.org/project/mmrotate/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)

MMRotate is an open source rotating object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://open-mmlab.github.io/) project.

The master branch works with **PyTorch 1.6+**.
The compatibility to earlier versions of PyTorch is not fully tested.

Documentation: https://mmrotate.readthedocs.io/en/latest/.

https://user-images.githubusercontent.com/10410257/154433305-416d129b-60c8-44c7-9ebb-5ba106d3e9d5.MP4

### Major features
- **Support multiple angle representations**

  MMRotate provides three mainstream angle representations to meet different paper settings.

- **Modular Design**

  We decompose the rotation detection framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Strong baseline and State of the art**

  The toolbox provides strong baselines and state-of-the-art methods in rotation detection.

## License

This project is released under the [Apache 2.0 license](LICENSE).


## Model Zoo

Supported algorithms:

<details open>
<summary>Detection</summary>

- [x] [Rotated RetinaNet-OBB/HBB](configs/rotated_retinanet/README.md) (ICCV'2017)
- [x] [Rotated FasterRCNN-OBB](configs/rotated_faster_rcnn/README.md) (TPAMI'2017)
- [x] [Rotated RepPoints-OBB](configs/rotated_reppoints/README.md) (ICCV'2019)
- [x] [RoI Transformer](configs/roi_trans/README.md) (CVPR'2019)
- [x] [Gliding Vertex](configs/gliding_vertex/README.md) (TPAMI'2020)
- [x] [R<sup>3</sup>Det](configs/r3det/README.md) (AAAI'2021)
- [x] [S<sup>2</sup>A-Net](configs/s2anet/README.md) (TGRS'2021)
- [x] [ReDet](configs/redet/README.md) (CVPR'2021)
- [x] [Beyond Bounding-Box](configs/cfa/README.md) (CVPR'2021)
- [x] [Oriented R-CNN](configs/oriented_rcnn/README.md) (ICCV'2021)
- [x] [GWD](configs/gwd/README.md) (ICML'2021)
- [x] [KLD](configs/kld/README.md) (NeurIPS'2021)
- [x] [SASM](configs/sasm_reppoints/README.md) (AAAI2022)
- [x] [KFIoU](configs/kfiou/README.md) (arXiv)
- [x] [G-Rep](configs/g_reppoints/README.md) (stay tuned)
</details>

Please refer to [model_zoo](docs/en/model_zoo.md) for more details.

## Changelog


## Installation & Dataset Preparation

Please refer to [install.md](docs/en/install.md) for installation of MMRotate and [data preparation](tools/data/README.md) for dataset preparation.

## Getting Started

If you are new of rotation detection, you can start with [learn the basics](docs/en/intro.md).
If you are familiar with it, check out [getting_started.md](docs/en/get_started.md) for the basic usage of mmrotate.


Refer to the below tutorials to dive deeper:

- [Learn the Basics](docs/en/intro.md)
- [Config](docs/en/tutorials/customize_config.md)
- [Customize Dataset](docs/en/tutorials/customize_dataset.md)
- [Customize Model](docs/en/tutorials/customize_models.md)

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{mmrotate2022,
    title={MMRotate: A rotation detection benchmark using PyTorch},
    author={Zhou, Yue and Yang, Xue and Zhang, Gefan},
    journal= {arXiv preprint arXiv:xxxx.xxxx},
    year={2022}
}
```


## Contributing

We appreciate all contributions to improve MMRotate. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmrotate/blob/main/.github/CONTRIBUTING.md) in MMRotate for the contributing guideline.

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab Human Pose and Shape Estimation Toolbox and Benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning Toolbox and Benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotation detection toolbox and benchmark.
