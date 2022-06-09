# 教程 4: 自定义训练设置

## 自定义优化设置

### 自定义 Pytorch 支持的优化器

我们已经支持了全部 Pytorch 自带的优化器，唯一需要修改的就是配置文件中 `optimizer` 部分。
例如，如果您想使用 `ADAM` (注意如下操作可能会让模型表现下降)，可以使用如下修改：

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

为了修改模型训练的学习率，使用者仅需修改配置文件里 `optimizer` 的 `lr` 即可。
使用者可以参考 PyTorch 的 [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 直接设置参数。

### 自定义用户自己实现的优化器

#### 1. 定义一个新的优化器

一个自定义的优化器可以这样定义：

假如您想增加一个叫做 `MyOptimizer` 的优化器，它的参数分别有 `a`, `b`, 和 `c`。
您需要创建一个名为 `mmrotate/core/optimizer` 的新文件夹；然后参考如下代码段在 `mmrotate/core/optimizer/my_optimizer.py` 文件中实现新的优化器:

```python
from mmdet.core.optimizer.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

#### 2. 增加优化器到注册表 (registry)

为了能够使得上述添加的模块被 `mmrotate` 发现，需要先将该模块添加到主命名空间（main namespace）。

- 修改 `mmrotate/core/optimizer/__init__.py` 文件来导入该模块。

  新的被定义的模块应该被导入到 `mmrotate/core/optimizer/__init__.py` 中，这样注册表才会发现新的模块并添加它：

```python
from .my_optimizer import MyOptimizer
```

- 在配置文件中使用 `custom_imports` 来手动添加该模块

```python
custom_imports = dict(imports=['mmrotate.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

`mmrotate.core.optimizer.my_optimizer` 模块将会在程序开始被导入，并且 `MyOptimizer` 类将会自动注册。
需要注意只有包含 `MyOptimizer` 类的包 (package) 应当被导入。
而 `mmrotate.core.optimizer.my_optimizer.MyOptimizer` **不能** 被直接导入。

事实上，在这种导入方式下用户可以用完全不同的文件夹结构，只要这一模块的根目录已经被添加到 `PYTHONPATH` 里面。

#### 3. 在配置文件中指定优化器

之后您可以在配置文件的 `optimizer` 部分里面使用 `MyOptimizer`。
在配置文件里，优化器按照如下形式被定义在 `optimizer` 部分里：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

要使用用户自定义的优化器，这部分应该改成：

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### 自定义优化器的构造函数 (constructor)

有些模型的优化器可能有一些特别参数配置，例如批归一化层 (BatchNorm layers) 的权重衰减系数 (weight decay)。
用户可以通过自定义优化器的构造函数去微调这些细粒度参数。

```python
from mmcv.utils import build_from_cfg

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmrotate.utils import get_root_logger
from .my_optimizer import MyOptimizer


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(object):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

`mmcv` 默认的优化器构造函数实现可以参考 [这里](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/optimizer/default_constructor.py#L11) ，这也可以作为新的优化器构造函数的模板。

### 其他配置

优化器未实现的技巧应该通过修改优化器构造函数（如设置基于参数的学习率）或者钩子（hooks）去实现。我们列出一些常见的设置，它们可以稳定或加速模型的训练。
如果您有更多的设置，欢迎在 PR 和 issue 里面提出。

- __使用梯度裁剪 (gradient clip) 来稳定训练__:
  一些模型需要梯度裁剪来稳定训练过程。使用方式如下：

  ```python
  optimizer_config = dict(
      _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
  ```

  如果您的配置继承了已经设置了 `optimizer_config` 的基础配置（base config），你可能需要设置 `_delete_=True` 来覆盖不必要的配置参数。请参考 [配置文档](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html) 了解更多细节。

- __使用动量调度加速模型收敛__:
  我们支持动量规划器（Momentum scheduler），以实现根据学习率调节模型优化过程中的动量设置，这可以使模型以更快速度收敛。
  动量规划器经常与学习率规划器（LR scheduler）一起使用，例如下面的配置经常被用于 3D 检测模型训练中以加速收敛。更多细节请参考 [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327) 和 [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130)。

  ```python
  lr_config = dict(
      policy='cyclic',
      target_ratio=(10, 1e-4),
      cyclic_times=1,
      step_ratio_up=0.4,
  )
  momentum_config = dict(
      policy='cyclic',
      target_ratio=(0.85 / 0.95, 1),
      cyclic_times=1,
      step_ratio_up=0.4,
  )
  ```

## 自定义训练计划

默认地，我们使用 1x 计划（1x schedule）的步进学习率（step learning rate），这在 MMCV 中被称为 [`StepLRHook`](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L153)。
我们支持很多其他的学习率规划器，参考 [这里](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py) ，例如 `CosineAnnealing` 和 `Poly`。下面是一些例子：

- `Poly` :

  ```python
  lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
  ```

- `ConsineAnnealing` :

  ```python
  lr_config = dict(
      policy='CosineAnnealing',
      warmup='linear',
      warmup_iters=1000,
      warmup_ratio=1.0 / 10,
      min_lr_ratio=1e-5)
  ```

## 自定义工作流 (workflow)

工作流是一个专门定义运行顺序和轮数（epochs）的列表。
默认情况下它设置成：

```python
workflow = [('train', 1)]
```

这是指训练 1 个 epoch。
有时候用户可能想检查一些模型在验证集上的指标，如损失函数值（Loss）和准确性（Accuracy）。
在这种情况下，我们可以将工作流设置为：

```python
[('train', 1), ('val', 1)]
```

这样以来， 1 个 epoch 训练，1 个 epoch 验证将交替运行。

**注意**:

1. 模型参数在验证的阶段不会被自动更新。
2. 配置文件里的键值 `total_epochs` 仅控制训练的 epochs 数目，而不会影响验证工作流。
3. 工作流 `[('train', 1), ('val', 1)]` 和 `[('train', 1)]` 将不会改变 `EvalHook` 的行为，因为 `EvalHook` 被 `after_train_epoch`
   调用而且验证的工作流仅仅影响通过调用 `after_val_epoch` 的钩子 (hooks)。因此， `[('train', 1), ('val', 1)]` 和 `[('train', 1)]`
   的区别仅在于 runner 将在每次训练阶段（training epoch）结束后计算在验证集上的损失。

## 自定义钩 (hooks)

### 自定义用户自己实现的钩子（hooks）

#### 1. 实现一个新的钩子（hook）

在某些情况下，用户可能需要实现一个新的钩子。 MMRotate 支持训练中的自定义钩子。 因此，用户可以直接在 mmrotate 或其基于 mmdet 的代码库中实现钩子，并通过仅在训练中修改配置来使用钩子。
这里我们举一个例子：在 mmrotate 中创建一个新的钩子并在训练中使用它。

```python
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
```

用户需要根据钩子的功能指定钩子在训练各阶段中（ `before_run` , `after_run` , `before_epoch` , `after_epoch` , `before_iter` , `after_iter`）做什么。

#### 2. 注册新的钩子（hook）

接下来我们需要导入 `MyHook`。如果文件的路径是 `mmrotate/core/utils/my_hook.py` ，有两种方式导入：

- 修改 `mmrotate/core/utils/__init__.py` 文件来导入

  新定义的模块需要在 `mmrotate/core/utils/__init__.py` 导入，注册表才会发现并添加该模块：

```python
from .my_hook import MyHook
```

- 在配置文件中使用 `custom_imports` 来手动导入

```python
custom_imports = dict(imports=['mmrotate.core.utils.my_hook'], allow_failed_imports=False)
```

#### 3. 修改配置

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

您也可以通过配置键值 `priority` 为 `'NORMAL'` 或 `'HIGHEST'` 来设置钩子的优先级：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

默认地，钩子的优先级在注册时被设置为 `NORMAL`。

### 使用 MMCV 中实现的钩子 (hooks)

如果钩子已经在 MMCV 里实现了，您可以直接修改配置文件来使用钩子。

#### 4. 示例: `NumClassCheckHook`

我们实现了一个自定义的钩子 [NumClassCheckHook](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/utils.py) ，用来检验 head 中的 `num_classes` 是否与 `dataset` 中的 `CLASSSES` 长度匹配。

我们在 [default_runtime.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/default_runtime.py) 中对其进行设置。

```python
custom_hooks = [dict(type='NumClassCheckHook')]
```

### 修改默认运行挂钩

有一些常见的钩子并不通过 `custom_hooks` 注册，这些钩子包括：

- log_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

这些钩子中，只有记录器钩子（logger hook）是 `VERY_LOW` 优先级，其他钩子的优先级为 `NORMAL`。
前面提到的教程已经介绍了如何修改 `optimizer_config` , `momentum_config` 以及 `lr_config`。
这里我们介绍一下如何处理 `log_config` , `checkpoint_config` 以及 `evaluation`。

#### Checkpoint config

MMCV runner 将使用 `checkpoint_config` 来初始化 [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/hooks/checkpoint.py#L9)。

```python
checkpoint_config = dict(interval=1)
```

用户可以设置 `max_keep_ckpts` 来仅保存一小部分检查点（checkpoint）或者通过设置 `save_optimizer` 来决定是否保存优化器的状态字典 (state dict of optimizer)。更多使用参数的细节请参考 [这里](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)。

#### Log config

`log_config` 包裹了许多日志钩 (logger hooks) 而且能去设置间隔 (intervals)。现在 MMCV 支持 `WandbLoggerHook` ， `MlflowLoggerHook` 和 `TensorboardLoggerHook`。
详细的使用请参照 [文档](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook)。

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### Evaluation config

`evaluation` 的配置文件将被用来初始化 [`EvalHook`](https://github.com/open-mmlab/mmdetection/blob/7a404a2c000620d52156774a5025070d9e00d918/mmdet/core/evaluation/eval_hooks.py#L8)。
除了 `interval` 键，其他的像 `metric` 这样的参数将被传递给 `dataset.evaluate()`。

```python
evaluation = dict(interval=1, metric='bbox')
```
