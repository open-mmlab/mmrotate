# 自定义数据集 (待更新)

## 支持新的数据格式

要支持新的数据格式，您可以将它们转换为现有的格式（DOTA 格式）。您可以选择离线（在通过脚本训练之前）或在线（实施新数据集并在训练时进行转换）进行转换。
在 MMRotate 中，我们建议将数据转换为 DOTA 格式并离线进行转换，如此您只需在数据转换后修改 config 的数据标注路径和类别即可。

### 将新数据格式重构为现有格式

最简单的方法是将数据集转换为现有数据集格式 (DOTA) 。

DOTA 格式的注解 txt 文件：

```text
184 2875 193 2923 146 2932 137 2885 plane 0
66 2095 75 2142 21 2154 11 2107 plane 0
...
```

每行代表一个对象，并将其记录为一个 10 维数组 `A` 。

- `A[0:8]`: 多边形的格式 `(x1, y1, x2, y2, x3, y3, x4, y4)` 。
- `A[8]`: 类别
- `A[9]`: 困难

在数据预处理之后，用户可以通过两个步骤来训练具有现有格式（例如 DOTA 格式）的自定义新数据集：

1. 修改配置文件以使用自定义数据集。
2. 检查自定义数据集的标注。

下面给出两个例子展示上述两个步骤，它使用一个自定义的 5 类 COCO 格式的数据集来训练一个现有的 Cascade Mask R-CNN R50-FPN 检测器。

#### 1. 修改配置文件以使用自定义数据集

配置文件的修改主要涉及两个方面:

1. `data` 部分。具体来说，您需要在 `data.train`, `data.val` 和 `data.test` 中显式添加 classes 字段。

2. `data` 属性变量。具体来说，特别是您需要在 `data.train`, `data.val` 和  `data.test` 中添加 classes 字段。

3. `model` 部分中的 ` num_classes`  属性变量。特别是将所有 num_classes 的默认值（例如 COCO 中的 80）覆盖到您的类别编号中。

在 `configs/my_custom_config.py` :

```python

# 新配置继承了基础配置用于突出显示必要的修改
_base_ = './rotated_retinanet_hbb_r50_fpn_1x_dota_oc'

# 1. 数据集的设置
dataset_type = 'DOTADataset'
classes = ('a', 'b', 'c', 'd', 'e')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,

        # 注意将你的类名添加到字段 `classes`
        classes=classes,
        ann_file='path/to/your/train/annotation_data',
        img_prefix='path/to/your/train/image_data'),
    val=dict(
        type=dataset_type,

        # 注意将你的类名添加到字段 `classes`
        classes=classes,
        ann_file='path/to/your/val/annotation_data',
        img_prefix='path/to/your/val/image_data'),
    test=dict(
        type=dataset_type,

        # 注意将你的类名添加到字段 `classes`
        classes=classes,
        ann_file='path/to/your/test/annotation_data',
        img_prefix='path/to/your/test/image_data'))

# 2. 模型设置
model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        # 显式将所有 `num_classes` 字段从 15 重写为 5。。
        num_classes=15))
```

#### 2. 查看自定义数据集的标注

假设您的自定义数据集是 DOTA 格式，请确保您在自定义数据集中具有正确的标注：

- 配置文件中的 `classes` 字段应该与 txt 标注的 `A[8]` 保持完全相同的元素和相同的顺序。
  MMRotate 会自动的将 `categories` 中不连续的 `id` 映射到连续的标签索引中，所以在 `categories` 中 `name` 的字符串顺序会影响标签索引的顺序。同时，配置文件中 `classes` 的字符串顺序也会影响预测边界框可视化过程中的标签文本信息。

## 通过封装器自定义数据集

MMRotate 还支持许多数据集封装器对数据集进行混合或修改数据集的分布以进行训练。目前它支持三个数据集封装器，如下所示：

- `RepeatDataset`: 简单地重复整个数据集。
- `ClassBalancedDataset`: 以类平衡的方式重复数据集。
- `ConcatDataset`: 拼接数据集。

### 重复数据集

我们使用 `RepeatDataset` 作为封装器来重复这个数据集。例如，假设原始数据集是 `Dataset_A`，我们就重复一遍这个数据集。配置信息如下所示：

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # 这是 Dataset_A 的原始配置信息
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### 类别平衡数据集

我们使用 `ClassBalancedDataset` 作为封装器，根据类别频率重复数据集。这个数据集的重复操作 `ClassBalancedDataset` 需要实例化函数 `self.get_cat_ids(idx)` 的支持。例如，`Dataset_A` 需要使用`oversample_thr=1e-3`，配置信息如下所示：

```python
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### 拼接数据集

这里用三种方式对数据集进行拼接。

1. 如果要拼接的数据集属于同一类型且具有不同的标注文件，则可以通过如下所示的配置信息来拼接数据集：

   ````python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       pipeline=train_pipeline
   )
   ```如果拼接后的数据集用于测试或评估，我们这种方式是可以支持对每个数据集分别进行评估。如果要测试的整个拼接数据集，如下所示您可以直接通过设置 separate_eval=False 来实现。

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       separate_eval=False,
       pipeline=train_pipeline
   )
   ````

2. 如果您要拼接不同的数据集，您可以通过如下所示的方法对拼接数据集配置信息进行设置。

   ````python
   dataset_A_train = dict()
   dataset_B_train = dict()
   data = dict(
       imgs_per_gpu=2,
       workers_per_gpu=2,
       train = [
           dataset_A_train,
           dataset_B_train
       ],
       val = dataset_A_val,
       test = dataset_A_test
       )
   ```如果拼接后的数据集用于测试或评估，这种方式还支持对每个数据集分别进行评估。

   ````

3. 我们也支持如下所示的方法对 `ConcatDataset` 进行明确的定义。

   ````python
   dataset_A_val = dict()
   dataset_B_val = dict()
   data = dict(
       imgs_per_gpu=2,
       workers_per_gpu=2,
       train=dataset_A_train,
       val=dict(
           type='ConcatDataset',
           datasets=[dataset_A_val, dataset_B_val],
           separate_eval=False))
   ```这种方式允许用户通过设置 `separate_eval=False` 将所有数据集转为单个数据集进行评估。

   ````

**笔记:**

1. 假设数据集在评估期间使用 `self.data_infos`，就要把选项设置为 `separate_eval=False`。因为 COCO 数据集不完全依赖 `self.data_infos` 进行评估，所以 COCO 数据集并不支持这种设置操作。没有在组合不同类型的数据集并对其进行整体评估的场景进行测试，因此我们不建议使用这样的操作。
2. 不支持评估 `ClassBalancedDataset` 和 `RepeatDataset`，所以也不支持评估这些类型的串联组合后的数据集。

一个更复杂的例子，分别将 `Dataset_A` 和 `Dataset_B` 重复 N 次和 M 次，然后将重复的数据集连接起来，如下所示。

```python
dataset_A_train = dict(
    type='RepeatDataset',
    times=N,
    dataset=dict(
        type='Dataset_A',
        ...
        pipeline=train_pipeline
    )
)
dataset_A_val = dict(
    ...
    pipeline=test_pipeline
)
dataset_A_test = dict(
    ...
    pipeline=test_pipeline
)
dataset_B_train = dict(
    type='RepeatDataset',
    times=M,
    dataset=dict(
        type='Dataset_B',
        ...
        pipeline=train_pipeline
    )
)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_A_train,
        dataset_B_train
    ],
    val = dataset_A_val,
    test = dataset_A_test
)
```
