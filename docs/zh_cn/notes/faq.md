# 常见问题解答 (待更新)

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。如果您无法在此获得帮助，请使用 [issue模板](https://github.com/open-mmlab/mmdetection/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) 创建问题，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## MMCV 安装相关

- MMCV 与 MMDetection 的兼容问题: "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

MMRotate 和 MMCV, MMDet 版本兼容性如下所示，需要安装正确的版本以避免安装出现问题。

| MMRotate | MMEngine                    | MMCV                       | MMDetection                 |
| -------- | --------------------------- | -------------------------- | --------------------------- |
| dev-1.x  | 0.6.0 \<= mmengine \< 1.0.0 | 2.0.0rc4 \<= mmcv \< 2.1.0 | 3.0.0rc6 \<= mmdet \< 3.2.0 |
| 1.0.0rc1 | 0.1.0 \<= mmengine \< 1.0.0 | 2.0.0rc2 \<= mmcv \< 2.1.0 | 3.0.0rc5 \<= mmdet \< 3.1.0 |
| 1.0.0rc0 | 0.1.0 \<= mmengine \< 1.0.0 | 2.0.0rc2 \<= mmcv \< 2.1.0 | 3.0.0rc2 \<= mmdet \< 3.1.0 |

**注意：**

1. 如果你希望安装 mmrotate-v0.x, MMRotate 和 MMCV 版本兼容表可以在 [这里](https://mmrotate.readthedocs.io/en/stable/faq.html#installation) 找到，请选择合适的版本避免安装错误。
2. 在 MMCV-v2.x 中，`mmcv-full` 改名为 `mmcv`，如果你想安装不包含 CUDA 算子的版本，可以选择安装 MMCV 精简版 `mmcv-lite`。

- "No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'".

  原因是安装了 `mmcv-lite` 而不是 `mmcv`。

  1. `pip uninstall mmcv-lite` 卸载安装的 `mmcv-lite`

  2. 安装 `mmcv` 根据 [安装说明](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)。

## PyTorch/CUDA 环境相关

- "RTX 30 series card fails when building MMCV or MMDet"

  1. 常见报错信息为 `nvcc fatal: Unsupported gpu architecture 'compute_86'` 意思是你的编译器应该为 sm_86 进行优化，例如， 英伟达 30 系列的显卡，但这样的优化 CUDA toolkit 11.0 并不支持。
     此解决方案通过添加 `MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e .` 来修改编译标志，这告诉编译器 `nvcc` 为 **sm_80** 进行优化，例如 Nvidia A100，尽管 A100 不同于 30 系列的显卡，但他们使用相似的图灵架构。这种解决方案可能会丧失一些性能但的确有效。
  2. PyTorch 开发者已经在 [pytorch/pytorch#47585](https://github.com/pytorch/pytorch/pull/47585) 更新了 PyTorch 默认的编译标志，所以使用 PyTorch-nightly 可能也能解决这个问题，但是我们对此并没有验证这种方式是否有效。

- "invalid device function" or "no kernel image is available for execution".

  1. 检查您的 cuda 运行时版本(一般在 `/usr/local/`)、指令 `nvcc --version` 显示的版本以及 `conda list cudatoolkit` 指令显式的版本是否匹配。
  2. 通过运行 `python mmdet/utils/collect_env.py` 来检查是否为当前的GPU架构编译了正确的 PyTorch、torchvision 和 MMCV，你可能需要设置 `TORCH_CUDA_ARCH_LIST` 来重新安装 MMCV。可以参考 [GPU 架构表](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)，即通过运行 `TORCH_CUDA_ARCH_LIST=7.0 pip install mmcv-full` 为 Volta GPU 编译 MMCV。这种架构不匹配的问题一般会出现在使用一些旧型号的 GPU 时候，例如，Tesla K80。
  3. 检查运行环境是否与 mmcv/mmdet 编译时相同，例如，您可能使用 CUDA 10.0 编译 MMCV，但在 CUDA 9.0 环境中运行它。

- "undefined symbol" or "cannot open xxx.so".

  1. 如果这些 symbols 属于 CUDA/C++ (例如，libcudart.so 或者 GLIBCXX)，检查 CUDA/GCC 运行时环境是否与编译 MMCV 的一致。例如使用 `python mmdet/utils/collect_env.py` 检查 `"MMCV Compiler"`/`"MMCV CUDA Compiler"` 是否和 `"GCC"`/`"CUDA_HOME"` 一致。
  2. 如果这些 symbols 属于 PyTorch，(例如，symbols containing caffe、aten 和 TH)， 检查当前 PyTorch 版本是否与编译 MMCV 的版本一致。
  3. 运行 `python mmdet/utils/collect_env.py` 检查 PyTorch、torchvision、MMCV 等的编译环境与运行环境一致。

- "setuptools.sandbox.UnpickleableException: DistutilsSetupError("each element of 'ext_modules' option must be an Extension instance or 2-tuple")"

  1. 如果你在使用 miniconda 而不是 anaconda，检查是否正确的安装了 Cython 如 [#3379](https://github.com/open-mmlab/mmdetection/issues/3379)。您需要先手动安装 Cpython 然后运命令 `pip install -r requirements.txt`。
  2. 检查环境中的 `setuptools`、`Cython` 和 `PyTorch` 相互之间版本是否匹配。

- "Segmentation fault".

  1. 检查 GCC 的版本并使用 GCC 5.4，通常是因为 PyTorch 版本与 GCC 版本不匹配 （例如，对于 Pytorch GCC \< 4.9)，我们推荐用户使用 GCC 5.4，我们也不推荐使用 GCC 5.5， 因为有反馈 GCC 5.5 会导致 "segmentation fault" 并且切换到 GCC 5.4 就可以解决问题。

  2. 检查是是否 PyTorch 被正确的安装并可以使用 CUDA 算子，例如在终端中键入如下的指令。

     ```shell
     python -c 'import torch; print(torch.cuda.is_available())'
     ```

     并判断是否是否返回 True。

  3. 如果 `torch` 的安装是正确的，检查是否正确编译了 MMCV。

     ```shell
     python -c 'import mmcv; import mmcv.ops'
     ```

     如果 MMCV 被正确的安装了，那么上面的两条指令不会有问题。

  4. 如果 MMCV 与 PyTorch 都被正确安装了，则使用 `ipdb`、`pdb` 设置断点，直接查找哪一部分的代码导致了 `segmentation fault`。

## E2CNN

- "ImportError: cannot import name 'container_bacs' from 'torch.\_six'"

  1. 这是因为 `container_abcs` 在 PyTorch 1.9 之后被移除.

  2. 将文件 `python3.7/site-packages/e2cnn/nn/modules/module_list.py` 中的

     ```shell
     from torch.six import container_abcs
     ```

     替换成

     ```shell
     TORCH_MAJOR = int(torch.__version__.split('.')[0])
     TORCH_MINOR = int(torch.__version__.split('.')[1])
     if TORCH_MAJOR ==1 and TORCH_MINOR < 8:
         from torch.six import container_abcs
     else:
         import collections.abs as container_abcs
     ```

  3. 或者降低 Pytorch 的版本。

## Training 相关

- "Loss goes Nan"

  1. 检查数据的标注是否正常，长或宽为 0 的框可能会导致回归 loss 变为 nan，一些小尺寸（宽度或高度小于 1）的框在数据增强（例如，instaboost）后也会导致此问题。因此，可以检查标注并过滤掉那些特别小甚至面积为 0 的框，并关闭一些可能会导致 0 面积框出现数据增强。
  2. 降低学习率：由于某些原因，例如 batch size 大小的变化，导致当前学习率可能太大。您可以降低为可以稳定训练模型的值。
  3. 延长 warm up 的时间：一些模型在训练初始时对学习率很敏感，您可以把 `warmup_iters` 从 500 更改为 1000 或 2000。
  4. 添加 gradient clipping: 一些模型需要梯度裁剪来稳定训练过程。默认的 `grad_clip` 是 `None`，你可以在 config 设置 `optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))`。 如果你的 config 没有继承任何包含 `optimizer_config=dict(grad_clip=None)`，你可以直接设置 `optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))`。

- "GPU out of memory"

  1. 存在大量 ground truth boxes 或者大量 anchor 的场景，可能在 assigner 会 OOM。您可以在 assigner 的配置中设置 `gpu_assign_thr=N`，这样当超过 N 个 GT boxes 时，assigner 会通过 CPU 计算 IoU。
  2. 在 backbone 中设置 `with_cp=True`。这使用 PyTorch 中的 `sublinear strategy` 来降低 backbone 占用的 GPU 显存。
  3. 通过在配置文件中设置 `fp16 = dict(loss_scale='dynamic')` 来尝试混合精度训练。

- "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"

  1. 错误表明，您的模块有没用于产生损失的参数，这种现象可能是由于在 DDP 模式下运行代码中的不同分支造成的。
  2. 您可以在配置中设置 `find_unused_parameters = True` 来解决上述问题，或者手动查找那些未使用的参数。

## Evaluation 相关

- 使用 COCO Dataset 的测评接口时，测评结果中 AP 或者 AR = -1。
  1. 根据 COCO 数据集的定义，一张图像中的中等物体与小物体面积的阈值分别为 9216（96\*96）与 1024（32\*32）。
  2. 如果在某个区间没有物体即 GT，AP 与 AR 将被设置为 -1。
