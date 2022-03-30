<div align="center">
  <img src="resources/mmrotate-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![Documentation](https://readthedocs.org/projects/mmrotate/badge/?version=latest)](https://mmrotate.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmrotate/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrotate/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrotate/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrotate)
[![PyPI](https://img.shields.io/pypi/v/mmrotate)](https://pypi.org/project/mmrotate/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)

[📘文档](https://mmrotate.readthedocs.io/en/latest/) |
[🛠️安装](https://mmrotate.readthedocs.io/en/latest/install.html) |
[👀模型库](docs/en/model_zoo.md) |
[🤔报告问题](https://github.com/open-mmlab/mmrotate/issues/new/choose)
</div>

## 介绍

[English](./README.md) | 简体中文

MMRotate 是一款基于 PyTorch 的旋转框检测的开源工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.6 以上**的版本。

https://user-images.githubusercontent.com/10410257/154433305-416d129b-60c8-44c7-9ebb-5ba106d3e9d5.MP4

<details open>
<summary><b>主要特性</b></summary>

* **支持多种角度表示法**

  MMRotate 提供了三种主流的角度表示法以满足不同论文的配置。

* **模块化设计**

  MMRotate 将旋转框检测任务解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的旋转框检测算法模型。

* **强大的基准模型与SOTA**

  MMRotate 提供了旋转框检测任务中最先进的算法和强大的基准模型.

</details>

## 更新日志

最新的 **0.1.1** 版本已经在 2022.03.14 发布:

- 为初学者添加了 [Colab 教程](demo/MMRotate_Tutorial.ipynb)
- 支持了[大图推理](deom/huge_image_demo.py)
- 支持了 HRSC 遥感数据集
- 支持了混合精度训练
- 添加了推理速度[统计工具](tools/analysis_tools/benchmark.py)
- 添加了混淆矩阵[分析工具](tools/analysis_tools/confusion_matrix.py).

如果想了解更多版本更新细节和历史信息，请阅读[更新日志](docs/en/changelog.md)。

## 安装

MMRotate 依赖 [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection)
，以下是安装的简要步骤。
更详细的安装指南请参考 [install.md](docs/zh_cn/install.md)。

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip3 install -e .
```

## 教程

请参考 [get_started.md](docs/zh_cn/get_started.md) 了解 MMRotate 的基本使用。
我们为初学者提供了 [colab 教程](demo/MMRotate_Tutorial.ipynb)。
MMRotate 也提供了其他更详细的教程:

* [学习基础知识](docs/zh_cn/intro.md)
* [配置文件](docs/zh_cn/tutorials/customize_config.md)
* [添加数据集](docs/zh_cn/tutorials/customize_dataset.md)
* [添加新模型](docs/zh_cn/tutorials/customize_models.md)。

## 模型库

各个模型的结果和设置都可以在对应的 config（配置）目录下的 *README.md* 中查看。
整体的概况也可也在 [模型库](docs/zh_cn/model_zoo.md) 页面中查看。

<details open>
<summary><b>支持的算法</b></summary>

* [x] [Rotated RetinaNet-OBB/HBB](configs/rotated_retinanet/README.md) (ICCV'2017)
* [x] [Rotated FasterRCNN-OBB](configs/rotated_faster_rcnn/README.md) (TPAMI'2017)
* [x] [Rotated RepPoints-OBB](configs/rotated_reppoints/README.md) (ICCV'2019)
* [x] [RoI Transformer](configs/roi_trans/README.md) (CVPR'2019)
* [x] [Gliding Vertex](configs/gliding_vertex/README.md) (TPAMI'2020)
* [x] [R<sup>3</sup>Det](configs/r3det/README.md) (AAAI'2021)
* [x] [S<sup>2</sup>A-Net](configs/s2anet/README.md) (TGRS'2021)
* [x] [ReDet](configs/redet/README.md) (CVPR'2021)
* [x] [Beyond Bounding-Box](configs/cfa/README.md) (CVPR'2021)
* [x] [Oriented R-CNN](configs/oriented_rcnn/README.md) (ICCV'2021)
* [x] [GWD](configs/gwd/README.md) (ICML'2021)
* [x] [KLD](configs/kld/README.md) (NeurIPS'2021)
* [x] [SASM](configs/sasm_reppoints/README.md) (AAAI'2022)
* [x] [KFIoU](configs/kfiou/README.md) (arXiv)
* [x] [G-Rep](configs/g_reppoints/README.md) (stay tuned)

</details>

### 模型需求

我们将跟进学界的最新进展，并支持更多算法和框架。如果您对 MMRotate 有任何功能需求，请随时在 [MMRotate Roadmap](https://github.com/open-mmlab/mmrotate/issues/1) 中留言。

## 数据准备

请参考 [data_preparation.md](tools/data/README.md) 进行数据集准备。

## 常见问题

请参考 [FAQ](docs/en/faq.md) 了解其他用户的常见问题。

## 参与贡献

我们非常欢迎用户对于 MMRotate 做出的任何贡献，可以参考 [CONTRIBUTION.md](.github/CONTRIBUTING.md) 文件了解更多细节。

## 致谢

MMRotate 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。
我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果您觉得 MMRotate 对您的研究有所帮助，请考虑引用它：

```bibtex
@misc{mmrotate2022,
  title={MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author =       {Zhou, Yue and Yang, Xue and Zhang, Gefan and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and Zhang, Wenwei, and Chen, Kai},
  howpublished = {\url{https://github.com/open-mmlab/mmrotate}},
  year =         {2022}
}
```

## 许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源协议。

## OpenMMLab的其他项目

* [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
* [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
* [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
* [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
* [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
* [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
* [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
* [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
* [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
* [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
* [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
* [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
* [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
* [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
* [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
* [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
* [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，联络 OpenMMLab [官方微信小助手](/docs/en/imgs/wechat_assistant_qrcode.png)或加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=GJP18SjI)

<div align="center">
<img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/zhihu_qrcode.jpg" height="400"><img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/wechat_qrcode.jpg" height="400"><img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/qq_group_qrcode.jpg" height="400">
</div>

我们会在 OpenMMLab 社区为大家

* 📢 分享 AI 框架的前沿核心技术
* 💻 解读 PyTorch 常用模块源码
* 📰 发布 OpenMMLab 的相关新闻
* 🚀 介绍 OpenMMLab 开发的前沿算法
* 🏃 获取更高效的问题答疑和意见反馈
* 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
