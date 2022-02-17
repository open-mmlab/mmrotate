## 依赖

- Linux & Windows
- Python 3.7+
- PyTorch 1.6+
- CUDA 9.2+
- GCC 5+
- [mmcv](https://mmcv.readthedocs.io/en/latest/#installation) 1.4.5+
- [mmdet](https://mmdetection.readthedocs.io/en/latest/#installation) 2.19.0+


MMRotate 和 MMCV, MMDet 版本兼容性如下所示，需要安装正确的版本以避免安装出现问题。

| MMRotate 版本   |    MMCV 版本   |      MMDetection 版本     |
|:-------------------:|:-----------------:|:---------------------------------:|
| master              | mmcv-full>=1.4.5 |      mmdet >= 2.19.0               |

**注意：**如果已经安装了 mmcv，首先需要使用 `pip uninstall mmcv` 卸载已安装的 mmcv，如果同时安装了 mmcv 和 mmcv-full，将会报 `ModuleNotFoundError` 错误。

## 安装流程

### 准备环境

1. 使用 conda 新建虚拟环境，并进入该虚拟环境；

    ```shell
    conda create -n openmmlab python=3.7 -y
    conda activate openmmlab
    ```

2. 基于 [PyTorch 官网](https://pytorch.org/)安装 PyTorch 和 torchvision，例如：

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

   **注意**：需要确保 CUDA 的编译版本和运行版本匹配。可以在 [PyTorch 官网](https://pytorch.org/)查看预编译包所支持的 CUDA 版本。

   `例 1` 例如在 `/usr/local/cuda` 下安装了 CUDA 10.1， 并想安装 PyTorch 1.7，则需要安装支持 CUDA 10.1 的预构建 PyTorch：

    ```shell
    conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch
    ```


### 安装 MMRotate

我们建议使用 [MIM](https://github.com/open-mmlab/mim) 来安装 MMRotate：

``` shell
pip install openmim
mim install mmrotate
```
MIM 能够自动地安装 OpenMMLab 的项目以及对应的依赖包。


或者，可以手动安装 MMRotate：

1. 安装 mmcv-full，我们建议使用预构建包来安装：

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    需要把命令行中的 `{cu_version}` 和 `{torch_version}` 替换成对应的版本。例如：在 CUDA 11 和 PyTorch 1.7.0 的环境下，可以使用下面命令安装最新版本的 MMCV：

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
    ```

    请参考 [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) 获取不同版本的 MMCV 所兼容的的不同的 PyTorch 和 CUDA 版本。同时，也可以通过以下命令行从源码编译 MMCV：

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # 安装好 mmcv-full
    cd ..
    ```

    或者，可以直接使用命令行安装：

    ```shell
    pip install mmcv-full
    ```

2. 安装 MMDetection.

    你可以直接通过如下命令从 pip 安装使用 mmdetection：

    ```shell
    pip install mmdet
    ```

3. 安装 MMRotate.

    你可以直接通过如下命令从 pip 安装使用 mmrotate：

    ```shell
    pip install mmrotate
    ```

    或者从 git 仓库编译源码：

    ```shell
    git clone https://github.com/open-mmlab/mmrotate.git
    cd mmrotate
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"

**Note:**

(1) 按照上述说明，MMDetection 安装在 `dev` 模式下，因此在本地对代码做的任何修改都会生效，无需重新安装；

(2) 如果希望使用 `opencv-python-headless` 而不是 `opencv-python`， 可以在安装 MMCV 之前安装；

(3) 一些安装依赖是可以选择的。例如只需要安装最低运行要求的版本，则可以使用 `pip install -v -e .` 命令。如果希望使用可选择的像 `albumentations` 和 `imagecorruptions` 这种依赖项，可以使用 `pip install -r requirements/optional.txt ` 进行手动安装，或者在使用 `pip` 时指定所需的附加功能（例如 `pip install -v -e .[optional]`），支持附加功能的有效键值包括  `all`、`tests`、`build` 以及 `optional` 。


### 另一种选择： Docker 镜像

我们提供了 [Dockerfile](https://github.com/open-mmlab/mmrotate/tree/main/docker/Dockerfile) to build an image. Ensure that you are using [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# 基于 PyTorch 1.6, CUDA 10.1 生成镜像
docker build -t mmrotate docker/
```

运行命令：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmrotate/data mmrotate
```

### 从零开始设置脚本

假设当前已经成功安装 CUDA 10.1，这里提供了一个完整的基于 conda 安装 MMDetection 的脚本：

```shell
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch

# 安装最新版本的 mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

# 安装 mmdetection
pip install mmdet

# 安装 mmrotate
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```


## 验证

为了验证是否正确安装了 MMRotate 和所需的环境，我们可以运行示例的 Python 代码在示例图像进行推理：

具体的细节可以参考 [demo](https://github.com/open-mmlab/mmrotate/tree/main/demo)。
如果成功安装 MMRotate，则上面的代码可以完整地运行。

## 准备数据集
具体的细节可以参考 [准备数据](https://github.com/open-mmlab/mmrotate/tree/main/tools/data) 下载并组织数据集。
