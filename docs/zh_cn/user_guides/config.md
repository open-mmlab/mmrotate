# 学习配置文件 (待更新)

我们在配置文件中支持了继承和模块化，这便于进行各种实验。
如果需要检查配置文件，可以通过运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置。
mmrotate 是建立在 [mmdet](https://github.com/open-mmlab/mmdetection) 之上的，
因此强烈建议学习 [mmdet](https://mmdetection.readthedocs.io/en/latest/) 的基本知识。

## 通过脚本参数修改配置

当运行 `tools/train.py` 或者 `tools/test.py` 时，可以通过 `--cfg-options` 来修改配置。

- 更新字典链的配置

  可以按照原始配置文件中的 dict 键顺序地指定配置预选项。
  例如，使用 `--cfg-options model.backbone.norm_eval=False` 将模型主干网络中的所有 BN 模块都改为 `train` 模式。

- 更新配置列表中的键

  在配置文件里，一些字典型的配置被包含在列表中。例如，数据训练流程 `data.train.pipeline` 通常是一个列表，比如  `[dict(type='LoadImageFromFile'), ...]`。 如果需要将 `'LoadImageFromFile'` 改成 `'LoadImageFromWebcam'` ，需要写成下述形式： `--cfg-options data.train.pipeline.0.type=LoadImageFromWebcam`。

- 更新列表或元组的值

  如果要更新的值是列表或元组。例如，配置文件通常设置 `workflow=[('train', 1)]`，如果需要改变这个键，可以通过 `--cfg-options workflow="[(train,1),(val,1)]"` 来重新设置。需要注意，引号 " 是支持列表或元组数据类型所必需的，并且在指定值的引号内**不允许**有空格。

## 配置文件名称风格

我们遵循以下样式来命名配置文件。建议贡献者遵循相同的风格。

```text
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{dataset}_{data setting}_{angle version}
```

`{xxx}` 是被要求的文件 `[yyy]` 是可选的。

- `{model}`： 模型种类，例如 `rotated_faster_rcnn`, `rotated_retinanet` 等。
- `[model setting]`： 特定的模型，例如 `hbb` for `rotated_retinanet` 等。
- `{backbone}`： 主干网络种类例如 `r50` (ResNet-50), `swin_tiny` (SWIN-tiny) 。
- `{neck}`： Neck 模型的种类包括 `fpn`,  `refpn`。
- `[norm_setting]`： 默认使用 `bn` (Batch Normalization)，其他指定可以有 `gn` (Group Normalization)， `syncbn` (Synchronized Batch Normalization) 等。 `gn-head`/`gn-neck` 表示 GN 仅应用于网络的 Head 或 Neck， `gn-all` 表示 GN 用于整个模型， 例如主干网络、Neck 和 Head。
- `[misc]`： 模型中各式各样的设置/插件，例如 `dconv`、 `gcb`、 `attention`、`albu`、 `mstrain` 等。
- `[gpu x batch_per_gpu]`： GPU 数量和每个 GPU 的样本数，默认使用 `1xb2`。
- `{dataset}`：数据集，例如 `dota`。
- `{angle version}`：旋转定义方式，例如 `oc`, `le135` 或者 `le90`。

## RotatedRetinaNet 配置文件示例

为了帮助用户对 MMRotate 检测系统中的完整配置和模块有一个基本的了解
我们对使用 ResNet50 和 FPN 的 RotatedRetinaNet 的配置文件进行简要注释说明。更详细的用法和各个模块对应的替代方案，请参考 API 文档。

```python
angle_version = 'oc'  # 旋转定义方式
model = dict(
    type='RotatedRetinaNet',  # 检测器(detector)名称
    backbone=dict(  # 主干网络的配置文件
        type='ResNet',  # # 主干网络的类别
        depth=50,  # 主干网络的深度
        num_stages=4,  # 主干网络阶段(stages)的数目
        out_indices=(0, 1, 2, 3),  # 每个阶段产生的特征图输出的索引
        frozen_stages=1,  # 第一个阶段的权重被冻结
        zero_init_residual=False,  # 是否对残差块(resblocks)中的最后一个归一化层使用零初始化(zero init)让它们表现为自身
        norm_cfg=dict(  # 归一化层(norm layer)的配置项
            type='BN',  # 归一化层的类别，通常是 BN 或 GN
            requires_grad=True),  # 是否训练归一化里的 gamma 和 beta
        norm_eval=True,  # 是否冻结 BN 里的统计项
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # 加载通过 ImageNet 预训练的模型
    neck=dict(
        type='FPN',  # 检测器的 neck 是 FPN， 我们同样支持 'ReFPN'
        in_channels=[256, 512, 1024, 2048],  # 输入通道数，这与主干网络的输出通道一致
        out_channels=256,  # 金字塔特征图每一层的输出通道
        start_level=1,  # 用于构建特征金字塔的主干网络起始输入层索引值
        add_extra_convs='on_input',  # 决定是否在原始特征图之上添加卷积层
        num_outs=5),  # 决定输出多少个尺度的特征图(scales)
    bbox_head=dict(
        type='RotatedRetinaHead',# bbox_head 的类型是 'RRetinaHead'
        num_classes=15,  # 分类的类别数量
        in_channels=256,  # bbox head 输入通道数
        stacked_convs=4,  # head 卷积层的层数
        feat_channels=256,  # head 卷积层的特征通道
        assign_by_circumhbbox='oc',  # obb2hbb 的旋转定义方式
        anchor_generator=dict(  # 锚点(Anchor)生成器的配置
            type='RotatedAnchorGenerator',  # 锚点生成器类别
            octave_base_scale=4,  # RetinaNet 用于生成锚点的超参数，特征图 anchor 的基本尺度。值越大，所有 anchor 的尺度都会变大。
            scales_per_octave=3,  #  RetinaNet 用于生成锚点的超参数，每个特征图有3个尺度
            ratios=[1.0, 0.5, 2.0],  # 高度和宽度之间的比率
            strides=[8, 16, 32, 64, 128]),  # 锚生成器的步幅。这与 FPN 特征步幅一致。如果未设置 base_sizes，则当前步幅值将被视为 base_sizes。
        bbox_coder=dict(  # 在训练和测试期间对框进行编码和解码
            type='DeltaXYWHAOBBoxCoder',  # 框编码器的类别
            angle_range='oc',  # 框编码器的旋转定义方式
            norm_factor=None,  # 框编码器的范数
            edge_swap=False,  # 设置是否启用框编码器的边缘交换
            proj_xy=False,  # 设置是否启用框编码器的投影
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),  # 用于编码和解码框的目标均值
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),  # 用于编码和解码框的标准差
        loss_cls=dict(  # 分类分支的损失函数配置
            type='FocalLoss',  # 分类分支的损失函数类型
            use_sigmoid=True,  #  是否使用 sigmoid
            gamma=2.0,  # Focal Loss 用于解决难易不均衡的参数 gamma
            alpha=0.25,  # Focal Loss 用于解决样本数量不均衡的参数 alpha
            loss_weight=1.0),  # 分类分支的损失权重
        loss_bbox=dict(  # 回归分支的损失函数配置
            type='L1Loss',  # 回归分支的损失类型
            loss_weight=1.0)),  # 回归分支的损失权重
    train_cfg=dict(  # 训练超参数的配置
        assigner=dict(  # 分配器(assigner)的配置
            type='MaxIoUAssigner',  # 分配器的类型
            pos_iou_thr=0.5,  # IoU >= 0.5(阈值) 被视为正样本
            neg_iou_thr=0.4,  # IoU < 0.4(阈值) 被视为负样本
            min_pos_iou=0,  # 将框作为正样本的最小 IoU 阈值
            ignore_iof_thr=-1,  # 忽略 bbox 的 IoF 阈值
            iou_calculator=dict(type='RBboxOverlaps2D')),  # IoU 的计算器类型
        allowed_border=-1,  # 填充有效锚点(anchor)后允许的边框
        pos_weight=-1,  # 训练期间正样本的权重
        debug=False),  # 是否设置调试(debug)模式
    test_cfg=dict(  # 测试超参数的配置
        nms_pre=2000,  # NMS 前的 box 数
        min_bbox_size=0,  # box 允许的最小尺寸
        score_thr=0.05,  # bbox 的分数阈值
        nms=dict(iou_thr=0.1), # NMS 的阈值
        max_per_img=2000))  # 每张图像的最大检测次数
dataset_type = 'DOTADataset'  # 数据集类型，这将被用来定义数据集
data_root = '../datasets/split_1024_dota1_0/'  # 数据的根路径
img_norm_cfg = dict(  # 图像归一化配置，用来归一化输入的图像
    mean=[123.675, 116.28, 103.53],  # 预训练里用于预训练主干网络模型的平均值
    std=[58.395, 57.12, 57.375],  # 预训练里用于预训练主干网络模型的标准差
    to_rgb=True)  # 预训练里用于预训练主干网络的图像的通道顺序
train_pipeline = [  # 训练流程
    dict(type='LoadImageFromFile'),  # 第 1 个流程，从文件路径里加载图像
    dict(type='LoadAnnotations',  # 第 2 个流程，对于当前图像，加载它的注释信息
         with_bbox=True),  # 是否加载标注框(bounding box)， 目标检测需要设置为 True
    dict(type='RResize',  # 变化图像和其注释大小的数据增广的流程
         img_scale=(1024, 1024)),  # 图像的最大规模
    dict(type='RRandomFlip',  # 翻转图像和其注释大小的数据增广的流程
         flip_ratio=0.5,  # 翻转图像的概率
         version='oc'),  # 定义旋转的方式
    dict(
        type='Normalize',  # 归一化当前图像的数据增广的流程
        mean=[123.675, 116.28, 103.53],  # 这些键与 img_norm_cfg 一致，
        std=[58.395, 57.12, 57.375],  # 因为 img_norm_cfg 被用作参数
        to_rgb=True),
    dict(type='Pad',  # 填充当前图像到指定大小的数据增广的流程
         size_divisor=32),  # 填充图像可以被当前值整除
    dict(type='DefaultFormatBundle'),  # 流程里收集数据的默认格式包
    dict(type='Collect',  # 决定数据中哪些键应该传递给检测器的流程
         keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [ # 测试流程
    dict(type='LoadImageFromFile'),  # 第 1 个流程，从文件路径里加载图像
    dict(
        type='MultiScaleFlipAug',  # 封装测试时数据增广(test time augmentations)
        img_scale=(1024, 1024),  # 决定测试时可改变图像的最大规模。用于改变图像大小的流程
        flip=False,  # 测试时是否翻转图像
        transforms=[
            dict(type='RResize'),  # 使用改变图像大小的数据增广
            dict(
                type='Normalize',  # 归一化配置项，值来自 img_norm_cfg
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad',  # 将配置传递给可被 32 整除的图像
                 size_divisor=32),
            dict(type='DefaultFormatBundle'),  # 用于在管道中收集数据的默认格式包
            dict(type='Collect',  # 收集测试时必须的键的收集流程
                 keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,  # 单个 GPU 的 Batch size
    workers_per_gpu=2,  # 单个 GPU 分配的数据加载线程数
    train=dict(  # 训练数据集配置
        type='DOTADataset',  # 数据集的类别
        ann_file=
        '../datasets/split_1024_dota1_0/trainval/annfiles/',  # 注释文件路径
        img_prefix=
        '../datasets/split_1024_dota1_0/trainval/images/',  # 图片路径前缀
        pipeline=[  # 流程, 这是由之前创建的 train_pipeline 传递的
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(1024, 1024)),
            dict(type='RRandomFlip', flip_ratio=0.5, version='oc'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        version='oc'),
    val=dict(  # 验证数据集的配置
        type='DOTADataset',
        ann_file=
        '../datasets/split_1024_dota1_0/trainval/annfiles/',
        img_prefix=
        '../datasets/split_1024_dota1_0/trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='oc'),
    test=dict(  # 测试数据集配置，修改测试开发/测试(test-dev/test)提交的 ann_file
        type='DOTADataset',
        ann_file=
        '../datasets/split_1024_dota1_0/test/images/',
        img_prefix=
        '../datasets/split_1024_dota1_0/test/images/',
        pipeline=[  # 由之前创建的 test_pipeline 传递的流程
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='oc'))
evaluation = dict(  # evaluation hook 的配置
    interval=12,  # 验证的间隔
    metric='mAP')  # 验证期间使用的指标
optimizer = dict(  # 用于构建优化器的配置文件
    type='SGD',  # 优化器类型
    lr=0.0025,  # 优化器的学习率
    momentum=0.9,  # 动量(Momentum)
    weight_decay=0.0001)  # SGD 的衰减权重(weight decay)
optimizer_config = dict(  # optimizer hook 的配置文件
    grad_clip=dict(
        max_norm=35,
        norm_type=2))
lr_config = dict(  # 学习率调整配置，用于注册 LrUpdater hook
    policy='step',  # 调度流程(scheduler)的策略
    warmup='linear',  # 预热(warmup)策略，也支持 `exp` 和 `constant`
    warmup_iters=500,  # 预热的迭代次数
    warmup_ratio=0.3333333333333333,  # 用于预热的起始学习率的比率
    step=[8, 11])  # 衰减学习率的起止回合数
runner = dict(
    type='EpochBasedRunner',  # 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner)
    max_epochs=12) # runner 总回合(epoch)数， 对于 IterBasedRunner 使用 `max_iters`
checkpoint_config = dict(  # checkpoint hook 的配置文件
    interval=12)  # 保存的间隔是 12
log_config = dict(  # register logger hook 的配置文件
    interval=50,  # 打印日志的间隔
    hooks=[
        # dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
        dict(type='TextLoggerHook')
    ])  # 用于记录训练过程的记录器(logger)
dist_params = dict(backend='nccl')  # 用于设置分布式训练的参数，端口也同样可被设置
log_level = 'INFO'  # 日志的级别
load_from = None  # 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间
resume_from = None  # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train', 1)]  # runner 的工作流程，[('train', 1)] 表示只有一个工作流且工作流仅执行一次。根据 total_epochs 工作流训练 12 个回合(epoch)。
work_dir = './work_dirs/rotated_retinanet_hbb_r50_fpn_1x_dota_oc'  # 用于保存当前实验的模型检查点(checkpoints)和日志的目录
```

## 常见问题 (FAQ)

### 使用配置文件里的中间变量

配置文件里会使用一些中间变量，例如数据集里的 `train_pipeline`/`test_pipeline`。
值得注意的是，在修改子配置中的中间变量时，需要再次将中间变量传递到相应的字段中。
例如，我们想使用离线多尺度策略 (multi scale strategy)来训练 RoI-Trans。 `train_pipeline` 是我们想要修改的中间变量。

```python
_base_ = ['./roi-trans-le90_r50_fpn_1x_dota.py']

data_root = '../datasets/split_ms_dota1_0/'
angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    val=dict(
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    test=dict(
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/'))
```

我们首先定义新的 `train_pipeline`/`test_pipeline` 然后传递到 `data` 里。

同样的，如果我们想从 `SyncBN` 切换到 `BN` 或者 `MMSyncBN`，我们需要修改配置文件里的每一个 `norm_cfg`。

```python
_base_ = './roi-trans-le90_r50_fpn_1x_dota.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    ...)
```
