# 开始你的第一步

## 需要准备的依赖环境

在本小节中将会演示如何准备一个Pytorch的环境。

MMRotate可以在Linux和Windows系统工作，需要以下环境版本：Python 3.7+, CUDA 9.2+ and PyTorch 1.6+。

```{note}
如果您对于Pytorch很熟悉并且已经完成了其安装步骤，您可以跳过本部分内容直接查阅[安装](#安装)的部分。当然，如果您没有准备好这部分的安装，请按照以下流程进行准备。
```

**第一步：** 从 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 下载并且安装Miniconda。

**第二步：** 创建一个虚拟环境并且切换至该虚拟环境中。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**第三步：** 根据 [Pytorch的官方说明](https://pytorch.org/get-started/locally/) 安装Pytorch, 例如：

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

## 安装

我们强烈建议用户使用以下方式安装MMRotate，这是最方便的。当然，还有多种自定义的安装方式可供经验丰富者选择,您可以查阅 [自定义安装](#customize-installation) 来获取更多相关的帮助。

### 最佳的安装方式

**第一步：** 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv) ,并且我们建议使用 [MIM](https://github.com/open-mmlab/mim) 来完成安装。

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc2"
```

**第二步：** 安装 [MMDetection](https://github.com/open-mmlab/mmdetection) 。

```shell
mim install 'mmdet>=3.0.0rc2'
```

注意：如果您需要进行部分代码的修改与开发，请从源码获取MMDetection并且进行以下示例的安装：

```shell
git clone https://github.com/open-mmlab/mmdetection.git -b dev-3.x
# "-b dev-3.x" means checkout to the `dev-3.x` branch.
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**第三步：** 安装 MMRotate。

情形a：如果您需要对于MMRotate进行代码上的开发与修改，请从源码进行以下示例的安装：

```shell
git clone https://github.com/open-mmlab/mmrotate.git -b dev-1.x
# "-b dev-1.x" means checkout to the `dev-1.x` branch.
cd mmrotate
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

情形b：如果您只是使用MMRotate作为依赖项或者是第三方软件包，则直接使用pip进行以下示例的安装即可：

```shell
pip install mmrotate
```

### 验证安装是否正确

为了验证MMRotate是否正确地被安装到了您的环境中，我们提供了一些demo以供测试。

**第一步：** 为了进行测试的验证，我们需要下载config文件与checkpoint文件。

```shell
mim download mmrotate --config oriented-rcnn-le90_r50_fpn_1x_dota --dest .
```

下载过程可能会依据您的网络状况花费数秒或者更多的时间。当下载完成之后，您可以在您的当前目录下找到 `oriented-rcnn-le90_r50_fpn_1x_dota.py` 和 `oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth` 两个文件。

**第二步：** 使用推理demo完成验证

情形 (a)：如果您是通过源码方式安装的MMRotate，直接在命令行中运行以下代码即可：

```shell
python demo/image_demo.py demo/demo.jpg oriented-rcnn-le90_r50_fpn_1x_dota.py oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result.jpg
```

您将会在您的当前文件夹下看到一张 `result.jpg` 为文件名的图片,在该图片中完成了对于小轿车和公交车等物体的旋转锚定框的绘制。

情形 (b)： 如果您是使用pip的方式安装的MMRotate，打开您的python控制台并使用代码：

```python
from mmdet.apis import init_detector, inference_detector
import mmrotate

config_file = 'oriented-rcnn-le90_r50_fpn_1x_dota.py'
checkpoint_file = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
inference_detector(model, 'demo/demo.jpg')
```

您将会看到一系列数组列表被打印出来，这代表着被检测到的旋转锚定框。

### 自定义安装

#### CUDA 版本

安装Pytorch之前，您需要指定安装的CUDA版本。如果您不清楚您的CUDA版本的安装选择，我们建议：

- 对于基于安培架构的NVIDIA GPU，例如GeForce 30系列和NVIDIA A100，请使用CUDA 11。
- 对于其他的NVIDIA GPU，CUDA 11同样可以兼容，但是CUDA 10.2更加轻量并且更好被兼容。

请确保GPU的驱动程序满足最低的版本需求，您可以查阅 [表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) 以了解更多信息。

```{note}
如果您按照我们推荐的最佳的安装方式进行安装流程，安装CUDA的运行库就已经足够了，因为这不需要在本地进行CUDA代码的编译工作。但是如果您想从源码对于MMCV进行编译或者开发其他的CUDA算子，您需要从NVIDIA官网完整地安装CUDA工具包 [官网](https://developer.nvidia.com/cuda-downloads) 并且CUDA工具包的版本应该和您使用的Pytorch版本对应。例如您可以使用 'conda install' 命令来安装指定版本的CUDA工具包。
```

#### 不使用MIM安装MMCV

MMCV有一些C++和CUDA扩展的使用,因此MMCV与Pytorch有着复杂的依赖关系。MIM会自动处理这些依赖关系使整个安装过程更加简便，但是您也可以选择不使用MIM这个工具。

如果您想使用pip进行MMCV的安装，而不是MIM，请按照 [MMCV安装指南](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 。您需要根据Pytorch版本和CUDA版本手动指定 find-url。

举个例子，以下代码示例是在PyTorch 1.9.x于CUDA 10.2的环境下进行MMCV的安装：

```shell
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8/index.html
```

#### 在Google Colab上安装

[Google Colab](https://research.google.com/) 通常已经完成了Pytorch的安装，
因此我们只需要按照以下步骤完成MMCV和MMDetection的安装即可。

**第一步：** 使用  [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection) :

```shell
!pip3 install -U openmim
!mim install "mmcv>=2.0.0rc2"
!mim install 'mmdet>=3.0.0rc2'
```

**第二步：** 从源码安装MMRotate：

```shell
!git clone https://github.com/open-mmlab/mmrotate.git -b dev-1.x
%cd mmrotate
!pip install -e .
```

**第三步：** 验证安装是否正确

```python
import mmrotate
print(mmrotate.__version__)
# Example output: 1.x
```

```{note}
在Jupyter中，感叹号 `!` 用于调用外部可执行文件，而符号 `%cd` 是一个用于更改Python当前工作目录的 [魔术指令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) 。
```

#### 使用Docker安装MMRotate

我们同样提供了 [Dockerfile](https://github.com/open-mmlab/mmrotate/tree/main/docker/Dockerfile) 用于创建镜像。请确认您的 [docker version](https://docs.docker.com/engine/install/) >=19.03。

```shell
# build an image with PyTorch 1.6, CUDA 10.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmrotate docker/
```

然后运行以下指令：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmrotate/data mmrotate
```

### 常见问题

如果您在安装的过程中遇到了一些困难，可以查询 [FAQ](faq.md) 页面。如果还是不能解决您的问题，您可以在github中 [提交Issue](https://github.com/open-mmlab/mmrotate/issues/new/choose) 。
