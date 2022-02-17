<div align="center">
  <img src="resources/mmrotate-logo.png" width="500px"/>
</div>

## 介绍

[English](README.md) | 简体中文

[![Documentation](https://readthedocs.org/projects/mmrotate/badge/?version=latest)](https://mmrotate.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmrotate/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrotate/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrotate/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrotate)
[![PyPI](https://badge.fury.io/py/mmrotate.svg)](https://pypi.org/project/mmrotate/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)

MMRotate 是一款基于 PyTorch 的旋转框检测代码库，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.6 以上**的版本。

https://user-images.githubusercontent.com/10410257/154433305-416d129b-60c8-44c7-9ebb-5ba106d3e9d5.MP4

### 主要特性

- **支持多种角度表示法**

  MMRotate 提供了三种主流的角度表示法以满足不同论文的配置。

- **模块化设计**

  MMRotate 将旋转框检测任务解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的旋转框检测算法模型。

- **强大的基准模型与SOTA**

  MMRotate 提供了旋转框检测任务中最先进的算法和强大的基准模型.



## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)



## 基准测试和模型库

已支持的算法：

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
- [x] [SASM](configs/sasm_reppoints/README.md) (AAAI‘2022)
- [x] [KFIoU](configs/kfiou/README.md) (arXiv)
- [x] [G-Rep](configs/g_reppoints/README.md) (stay tuned)
</details>

请点击[模型库](docs/zh_cn/model_zoo.md)查看更多关于上述算法的详细信息。

## 更新记录

## 安装与准备数据集

请参考[安装文档](docs/zh_cn/install.md)进行安装和参考[数据准备](tools/data/README.md)准备数据集。

## 快速入门
如果初次了解旋转框检测，你可以从[基础介绍](docs/zh_cn/intro.md)开始了解旋转框检测的基本概念和 MMRotate 的框架。
如果对旋转框检测很熟悉，请参考[使用教程](docs/zh_cn/get_started.md)获取 MMRotate 的基本用法。

MMRotate 也提供了其他更详细的教程，包括：

- [学习基础知识](docs/zh_cn/intro.md)
- [配置文件](docs/zh_cn/tutorials/customize_config.md)
- [添加数据集](docs/zh_cn/tutorials/customize_dataset.md)
- [添加新模型](docs/zh_cn/tutorials/customize_models.md)。


## 贡献指南

我们感谢所有的贡献者为改进和提升 MMRotate 所作出的努力。请参考[贡献指南](https://github.com/open-mmlab/mmrotate/blob/main/.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。


## 致谢

MMRotate 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。

我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。


## 引用
如果您发现此项目对您的研究有用，请考虑引用：

```bibtex
@article{mmrotate2022,
    title={MMRotate: A Rotation Detection Benchmark using PyTorch},
    author={Zhou, Yue and Yang, Xue and Zhang, Gefan and et al.},
    journal= {arXiv preprint arXiv:xxxx.xxxx},
    year={2022}
}
```


## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准

## 欢迎加入 OpenMMLab 社区

 扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="resources/zhihu_qrcode.jpg" height="400" />  <img src="resources/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
