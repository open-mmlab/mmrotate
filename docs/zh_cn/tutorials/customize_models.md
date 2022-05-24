# 教程 3: 自定义模型

我们大致将模型组件分为了 5 种类型。

- 主干网络 (Backbone): 通常是一个全卷积网络 (FCN)，用来提取特征图，比如残差网络 (ResNet)。也可以是基于视觉 Transformer 的网络，比如 Swin Transformer 等。
- Neck: 主干网络和任务头 (Head) 之间的连接组件，比如 FPN, ReFPN。
- 任务头 (Head): 用于某种具体任务（比如边界框预测）的组件。
- 区域特征提取器 (Roi Extractor): 用于从特征图上提取区域特征的组件，比如 RoI Align Rotated。
- 损失 (loss): 任务头上用于计算损失函数的组件，比如 FocalLoss, GWDLoss, and KFIoULoss。

## 开发新的组件

### 添加新的主干网络

这里，我们以 MobileNet 为例来展示如何开发新组件。

#### 1. 定义一个新的主干网络（以 MobileNet 为例）

新建文件 `mmrotate/models/backbones/mobilenet.py`。

```python
import torch.nn as nn

from mmrotate.models.builder import ROTATED_BACKBONES


@ROTATED_BACKBONES.register_module()
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass
```

#### 2. 导入模块

你可以将下面的代码添加到 `mmrotate/models/backbones/__init__.py` 中：

```python
from .mobilenet import MobileNet
```

或者添加如下代码

```python
custom_imports = dict(
    imports=['mmrotate.models.backbones.mobilenet'],
    allow_failed_imports=False)
```

到配置文件中以避免修改原始代码。

#### 3. 在你的配置文件中使用该主干网络

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

### 添加新的 Neck

#### 1. 定义一个 Neck（以 PAFPN 为例）

新建文件 `mmrotate/models/necks/pafpn.py`。

```python
from mmrotate.models.builder import ROTATED_NECKS

@ROTATED_NECKS.register_module()
class PAFPN(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                start_level=0,
                end_level=-1,
                add_extra_convs=False):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 导入该模块

你可以添加下述代码到 `mmrotate/models/necks/__init__.py` 中

```python
from .pafpn import PAFPN
```

或者添加

```python
custom_imports = dict(
    imports=['mmrotate.models.necks.pafpn.py'],
    allow_failed_imports=False)
```

到配置文件中以避免修改原始代码。

#### 3. 修改配置文件

```python
neck=dict(
    type='PAFPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5)
```

### 添加新的 Head

这里，我们以 [Double Head R-CNN](https://arxiv.org/abs/1904.06493) 为例来展示如何添加一个新的 Head。

首先，添加一个新的 bbox head 到 `mmrotate/models/roi_heads/bbox_heads/double_bbox_head.py`。
Double Head R-CNN 在目标检测上实现了一个新的 bbox head。为了实现 bbox head，我们需要使用如下的新模块中三个函数。

```python
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.models.roi_heads.bbox_heads.bbox_head import BBoxHead

@ROTATED_HEADS.register_module()
class DoubleConvFCBBoxHead(BBoxHead):
    r"""Bbox head used in Double-Head R-CNN

                                      /-> cls
                  /-> shared convs ->
                                      \-> reg
    roi features
                                      /-> cls
                  \-> shared fc    ->
                                      \-> reg
    """  # noqa: W605

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHead, self).__init__(**kwargs)


    def forward(self, x_cls, x_reg):

```

然后，如有必要，我们需要实现一个新的 RoI Head。我们打算从 `StandardRoIHead` 继承出新的 `DoubleHeadRoIHead`。我们发现 `StandardRoIHead` 已经实现了下述函数。

```python
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmrotate.models.builder import ROTATED_HEADS, build_head, build_roi_extractor
from mmrotate.models.roi_heads.base_roi_head import BaseRoIHead
from mmrotate.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin


@ROTATED_HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head.
    """

    def init_assigner_sampler(self):

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):

    def forward_dummy(self, x, proposals):


    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

    def _bbox_forward(self, x, rois):

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""

```

Double Head 的修改主要在 \_bbox_forward 的逻辑中，且它从 `StandardRoIHead` 中继承了其他逻辑。
在 `mmrotate/models/roi_heads/double_roi_head.py` 中, 我们实现如下的新的 RoI Head:

```python
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.models.roi_heads.standard_roi_head import StandardRoIHead


@ROTATED_HEADS.register_module()
class DoubleHeadRoIHead(StandardRoIHead):
    """RoI head for Double Head RCNN

    https://arxiv.org/abs/1904.06493
    """

    def __init__(self, reg_roi_scale_factor, **kwargs):
        super(DoubleHeadRoIHead, self).__init__(**kwargs)
        self.reg_roi_scale_factor = reg_roi_scale_factor

    def _bbox_forward(self, x, rois):
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results
```

最后，用户需要把这个新模块添加到 `mmrotate/models/bbox_heads/__init__.py` 以及 `mmrotate/models/roi_heads/__init__.py` 中。 这样，注册机制就能找到并加载它们。

另外，用户也可以添加

```python
custom_imports=dict(
    imports=['mmrotate.models.roi_heads.double_roi_head', 'mmrotate.models.bbox_heads.double_bbox_head'])
```

到配置文件中来实现同样的目的。

### 添加新的损失

假设你想添加一个新的损失 `MyLoss` 用于边界框回归。
为了添加一个新的损失函数，用户需要在 `mmrotate/models/losses/my_loss.py` 中实现。
装饰器 `weighted_loss` 可以使损失每个部分加权。

```python
import torch
import torch.nn as nn

from mmrotate.models.builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

@ROTATED_LOSSES.register_module()
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
```

然后，用户需要把下面的代码加到 `mmrotate/models/losses/__init__.py` 中。

```python
from .my_loss import MyLoss, my_loss

```

或者，你可以添加：

```python
custom_imports=dict(
    imports=['mmrotate.models.losses.my_loss'])
```

到配置文件来实现相同的目的。

因为 MyLoss 是用于回归的，你需要在 Head 中修改 `loss_bbox` 字段：

```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0))
```
