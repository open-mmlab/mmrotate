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

[![PyPI](https://img.shields.io/pypi/v/mmrotate)](https://pypi.org/project/mmrotate)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmrotate.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmrotate/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrotate/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrotate/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrotate)
[![license](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)

[📘使用文档](https://mmrotate.readthedocs.io/zh_CN/1.x/) |
[🛠️安装教程](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html) |
[👀模型库](https://mmrotate.readthedocs.io/zh_CN/1.x/model_zoo.html) |
[🆕更新日志](https://mmrotate.readthedocs.io/en/1.x/notes/changelog.html) |
[🚀进行中的项目](https://github.com/open-mmlab/mmrotate/projects) |
[🤔报告问题](https://github.com/open-mmlab/mmrotate/issues/new/choose)

</div>

<div align="center">

[English](/README.md) | 简体中文

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## 介绍

MMRotate 是一款基于 PyTorch 的旋转框检测的开源工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.6 以上**的版本。

https://user-images.githubusercontent.com/10410257/154433305-416d129b-60c8-44c7-9ebb-5ba106d3e9d5.MP4

<details open>
<summary><b>主要特性</b></summary>

- **支持多种角度表示法**

  MMRotate 提供了三种主流的角度表示法以满足不同论文的配置。

- **模块化设计**

  MMRotate 将旋转框检测任务解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的旋转框检测算法模型。

- **强大的基准模型与SOTA**

  MMRotate 提供了旋转框检测任务中最先进的算法和强大的基准模型.

</details>

## 最新进展

### 亮点

我们很高兴向大家介绍我们在实时目标识别任务方面的最新成果 RTMDet，包含了一系列的全卷积单阶段检测模型。 RTMDet 不仅在从 tiny 到 extra-large 尺寸的目标检测模型上实现了最佳的参数量和精度的平衡，而且在实时实例分割和旋转目标检测任务上取得了最先进的成果。 更多细节请参阅[技术报告](https://arxiv.org/abs/2212.07784)。 预训练模型可以在[这里](configs/rotated_rtmdet)找到。

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

| Task                     | Dataset | AP                                   | FPS(TRT FP16 BS1 3090) |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
| Object Detection         | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

最新的 **v1.0.0rc1** 版本已经在 2022.12.30 发布:

- 支持了 [RTMDet](configs/rotated_rtmdet) 的旋转目标检测模型。RTMDet 的技术报告发布在了 [arxiv](https://arxiv.org/abs/2212.07784) 上。
- 支持了 [H2RBox](configs/h2rbox) 模型。H2RBox 的技术报告发布在了 [arxiv](https://arxiv.org/abs/2210.06742) 上。

## 安装

请参考[快速入门文档](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html)进行安装。

## 教程

请阅读[概述](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html)对 MMDetection 进行初步的了解。

为了帮助用户更进一步了解 MMDetection，我们准备了用户指南和进阶指南，请阅读我们的[文档](https://mmrotate.readthedocs.io/zh_CN/1.x/)：

- 用户指南
  - [训练 & 测试](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/index.html#train-test)
    - [学习配置文件](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/config.html)
    - [使用已有模型在标准数据集上进行推理](https://mmrotate.readthedocs.io/en/1.x/user_guides/inference.html)
    - [数据集准备](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/dataset_prepare.html)
    - [测试现有模型](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/train_test.html#test)
    - [在标准数据集上训练预定义的模型](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/train_test.html#train)
    - [提交测试结果](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/test_results_submission.html)
  - [实用工具](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/index.html#useful-tools)
- 进阶指南
  - [基础概念](https://mmrotate.readthedocs.io/zh_CN/1.x/advanced_guides/index.html#basic-concepts)
  - [组件定制](https://mmrotate.readthedocs.io/zh_CN/1.x/advanced_guides/index.html#component-customization)
  - [How to](https://mmrotate.readthedocs.io/zh_CN/1.x/advanced_guides/index.html#how-to)

我们提供了旋转检测的 colab 教程 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo/MMRotate_Tutorial.ipynb)。

若需要将0.x版本的代码迁移至新版，请参考[迁移文档](https://mmrotate.readthedocs.io/zh_CN/1.x/migration.html)。

## 模型库

各个模型的结果和设置都可以在对应的 config（配置）目录下的 *README.md* 中查看。
整体的概况也可也在 [模型库](docs/zh_cn/model_zoo.md) 页面中查看。

<details open>
<summary><b>支持的算法</b></summary>

- [x] [Rotated RetinaNet-OBB/HBB](configs/rotated_retinanet/README.md) (ICCV'2017)
- [x] [Rotated FasterRCNN-OBB](configs/rotated_faster_rcnn/README.md) (TPAMI'2017)
- [x] [Rotated RepPoints-OBB](configs/rotated_reppoints/README.md) (ICCV'2019)
- [x] [Rotated FCOS](configs/rotated_fcos/README.md) (ICCV'2019)
- [x] [RoI Transformer](configs/roi_trans/README.md) (CVPR'2019)
- [x] [Gliding Vertex](configs/gliding_vertex/README.md) (TPAMI'2020)
- [x] [Rotated ATSS-OBB](configs/rotated_atss/README.md) (CVPR'2020)
- [x] [CSL](configs/csl/README.md) (ECCV'2020)
- [x] [R<sup>3</sup>Det](configs/r3det/README.md) (AAAI'2021)
- [x] [S<sup>2</sup>A-Net](configs/s2anet/README.md) (TGRS'2021)
- [x] [ReDet](configs/redet/README.md) (CVPR'2021)
- [x] [Beyond Bounding-Box](configs/cfa/README.md) (CVPR'2021)
- [x] [Oriented R-CNN](configs/oriented_rcnn/README.md) (ICCV'2021)
- [x] [GWD](configs/gwd/README.md) (ICML'2021)
- [x] [KLD](configs/kld/README.md) (NeurIPS'2021)
- [x] [SASM](configs/sasm_reppoints/README.md) (AAAI'2022)
- [x] [Oriented RepPoints](configs/oriented_reppoints/README.md) (CVPR'2022)
- [x] [KFIoU](configs/kfiou/README.md) (ICLR'2023)
- [x] [H2RBox](configs/h2rbox/README.md) (ICLR'2023)
- [x] [PSC](configs/psc/README.md) (CVPR'2023)
- [x] [RTMDet](configs/rotated_rtmdet/README.md) (arXiv)
- [x] [H2RBox-v2](configs/h2rbox_v2/README.md) (arXiv)

</details>

## 数据准备

请参考 [data_preparation.md](tools/data/README.md) 进行数据集准备。

## 常见问题

请参考 [FAQ](docs/en/notes/faq.md) 了解其他用户的常见问题。

## 参与贡献

我们非常欢迎用户对于 MMRotate 做出的任何贡献，可以参考 [CONTRIBUTION.md](.github/CONTRIBUTING.md) 文件了解更多细节。

## 致谢

MMRotate 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。我们感谢 [SJTU 学生创新中心](https://www.si.sjtu.edu.cn/)在项目之初提供的大量计算资源。我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 MMRotate。

```bibtex
@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages = {7331–7334},
  numpages = {4},
  year={2022}
}
```

## 许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源协议。

## OpenMMLab的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab 深度学习预训练工具箱
- [MMagic](https://github.com/open-mmlab/mmagic): OpenMMLab 新一代人工智能内容生成（AIGC）工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架
- [MIM](https://github.com/open-mmlab/mim): OpenMMlab 项目、算法、模型的统一入口
- [MMEval](https://github.com/open-mmlab/mmeval): 统一开放的跨框架算法评测库
- [Playground](https://github.com/open-mmlab/playground): 收集和展示 OpenMMLab 相关的前沿、有趣的社区项目

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)或联络 OpenMMLab 官方微信小助手

<div align="center">
<img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/zhihu_qrcode.jpg" height="400"><img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/wechat_qrcode.jpg" height="400"><img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/qq_group_qrcode.jpg" height="400">
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
