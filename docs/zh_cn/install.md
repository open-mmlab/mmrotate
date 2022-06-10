## 依赖

在本节中，我们将演示如何使用PyTorch准备环境。

MMRotate 能够在 Linux 和 Windows 上运行。它依赖于 Python 3.7+, CUDA 9.2+ 和 PyTorch 1.6+。

```{note}
如果您对 PyTorch 有经验并且已经安装了它，只需跳过此部分并跳到 [下一节](#安装) 。否则，您可以按照以下步骤进行准备。
```

**第0步：** 从 [官网](https://docs.conda.io/en/latest/miniconda.html) 下载并安装 Miniconda。

**第1步：** 创建一个 conda 环境并激活它.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**第2步：** 根据 [官方说明](https://pytorch.org/get-started/locally/) 安装 PyTorch, 例如：

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

## 安装

我们建议用户按照我们的最佳实践安装 MMRotate。然而，整个过程是高度可定制的。有关详细信息，请参阅 [自定义安装](#%E8%87%AA%E5%AE%9A%E4%B9%89%E5%AE%89%E8%A3%85) 部分。

### 最佳实践

**第0步：** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection)

```shell
pip install -U openmim
mim install mmcv-full
mim install mmdet
```

**第1步：** 安装 MMRotate.

案例a：如果您直接开发并运行 mmrotate，请从源代码安装：

```shell
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
# "-v" 表示详细或更多输出
# "-e" 表示以可编辑模式安装项目，
# 因此，对代码进行的任何本地修改都将在不重新安装的情况下生效。
```

案例b：如果将 mmrotate 作为依赖项或第三方软件包，请使用 pip 安装它：

```shell
pip install mmrotate
```

### 验证

为了验证是否正确安装了 MMRotate，我们提供了一些示例代码来运行推理演示。

**第1步：** 我们需要下载配置文件和检查点文件。

```shell
mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest .
```

下载需要几秒钟或更长时间，具体取决于您的网络环境。当下载完成之后，您将会在当前文件夹下找到 `oriented_rcnn_r50_fpn_1x_dota_le90.py` 和 `oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth` 这两个文件。

**第2步：** 验证推理演示

选项（a）：如果从源代码安装 mmrotate，只需运行以下命令。

```shell
python demo/image_demo.py demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result.jpg
```

您将在当前目录下看到一张名为 `result.jpg` 的新图片，其中旋转边界框绘制在汽车、公共汽车等目标上。

选项（b）：如果使用 pip 安装 mmrotate，请打开 python 解释器并复制和粘贴以下代码。

```python
from mmdet.apis import init_detector, inference_detector
import mmrotate

config_file = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'
checkpoint_file = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
inference_detector(model, 'demo/demo.jpg')
```

您将看到打印的数组列表，表示检测到的旋转边界框。

### 自定义安装

#### CUDA 版本

安装 PyTorch 时，需要指定 CUDA 的版本。如果您不清楚选择哪一个，请遵循我们的建议：

- 对于基于安培架构的 NVIDIA GPU，如 GeForce 30 系列和 NVIDIA A100，必须使用 CUDA 11。
- 对于较旧的 NVIDIA GPU，CUDA 11 向后兼容，但 CUDA 10.2 更轻量并且具有更好的兼容性。

请确保 GPU 驱动程序满足最低版本要求。 请查询 [表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) 以获得更多信息。

```{note}
如果遵循我们的最佳实践，安装 CUDA 运行时库就足够了，因为不会在本地编译 CUDA 代码。但是，如果您希望从源代码处编译 MMCV 或开发其他 CUDA 算子，则需要从 NVIDIA 的 [网站](https://developer.nvidia.com/cuda-downloads) 安装完整的 CUDA 工具包，其版本应与 PyTorch 的 CUDA 版本匹配。例如使用 `conda install` 命令指定 cudatoolkit 的版本。
```

#### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此以复杂的方式依赖于 PyTorch。MIM 会自动解决此类依赖关系，并使安装更容易。然而，这不是必须的。

要使用 pip 而不是 MIM 安装 MMCV，请遵循 [MMCV 安装指南](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 。 这需要根据 PyTorch 版本及其 CUDA 版本手动指定 find-url。

例如, 以下命令安装了为 PyTorch 1.9.x 和 CUDA 10.2 构建的 mmcv-full。

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8/index.html
```

#### 在 Google Colab 安装

[Google Colab](https://research.google.com/) 通常已经安装了 PyTorch，
因此，我们只需要使用以下命令安装 MMCV 和 MMDetection。

**第1步：** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection) 。

```shell
!pip3 install -U openmim
!mim install mmcv-full
!mim install mmdet
```

**第2步：** 从源码安装 MMRotate。

```shell
!git clone https://github.com/open-mmlab/mmrotate.git
%cd mmrotate
!pip install -e .
```

**第3步：** 验证。

```python
import mmrotate
print(mmrotate.__version__)
# Example output: 0.3.1
```

```{note}
在Jupyter中，感叹号 `!` 用于调用外部可执行文件，`%cd` 是一个 [魔术命令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) 用于更改 Python 的当前工作目录。
```

#### 在 Docker 镜像中使用 MMRotate

我们提供了 [Dockerfile](https://github.com/open-mmlab/mmrotate/tree/main/docker/Dockerfile) 用于创建镜像。 请确保您的 [docker 版本](https://docs.docker.com/engine/install/) >=19.03。

```shell
# build an image with PyTorch 1.6, CUDA 10.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmrotate docker/
```

使用下列命令运行

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmrotate/data mmrotate
```

### 故障排除

如果您在安装过程中遇到一些问题，请先查看 [FAQ](faq.md) 页面。
如果没有找到解决方案，您可以在 GIthub 上 [提问](https://github.com/open-mmlab/mmrotate/issues/new/choose) 。
