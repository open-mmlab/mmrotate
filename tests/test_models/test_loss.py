# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import pytest
import torch

from mmrotate import digit_version
from mmrotate.models.losses import (BCConvexGIoULoss, ConvexGIoULoss, GDLoss,
                                    GDLoss_v1, KFLoss, KLDRepPointsLoss,
                                    RotatedIoULoss)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('loss_class',
                         [BCConvexGIoULoss, ConvexGIoULoss, KLDRepPointsLoss])
def test_convex_regression_losses(loss_class):
    """Tests convex regression losses.

    Args:
        loss_class (str): type of convex loss.
    """
    pred = torch.rand((10, 18)).cuda()
    target = torch.rand((10, 8)).cuda()
    weight = torch.rand((10, )).cuda()

    # Test loss forward
    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with weight
    loss = loss_class()(pred, target, weight)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = loss_class()(pred, target, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = loss_class()(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)


# @pytest.mark.skipif(
#     not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('loss_type',
                         ['gwd', 'kld', 'jd', 'kld_symmax', 'kld_symmin'])
def test_gaussian_regression_losses(loss_type):
    """Tests gaussian regression losses.

    Args:
        loss_class (str): type of gaussian loss.
    """
    pred = torch.rand((10, 5))
    target = torch.rand((10, 5))
    weight = torch.rand((10, 5))

    # Test loss forward with weight
    loss = GDLoss(loss_type)(pred, target, weight)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = GDLoss(loss_type)(pred, target, weight, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = GDLoss(loss_type)(pred, target, weight, avg_factor=10)
    assert isinstance(loss, torch.Tensor)


# @pytest.mark.skipif(
#     not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('loss_type', ['bcd', 'kld', 'gwd'])
def test_gaussian_v1_regression_losses(loss_type):
    """Tests gaussian regression losses v1.

    Args:
        loss_class (str): type of gaussian loss v1.
    """
    pred = torch.rand((10, 5))
    target = torch.rand((10, 5))
    weight = torch.rand((10, 5))

    # Test loss forward with weight
    loss = GDLoss_v1(loss_type)(pred, target, weight)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = GDLoss_v1(loss_type)(
        pred, target, weight, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = GDLoss_v1(loss_type)(pred, target, weight, avg_factor=10)
    assert isinstance(loss, torch.Tensor)


# @pytest.mark.skipif(
#     not torch.cuda.is_available(), reason='requires CUDA support')
def test_kfiou_regression_losses():
    """Tests kfiou regression loss."""
    pred = torch.rand((10, 5))
    target = torch.rand((10, 5))
    weight = torch.rand((10, 5))
    pred_decode = torch.rand((10, 5))
    targets_decode = torch.rand((10, 5))

    # Test loss forward with weight
    loss = KFLoss()(
        pred,
        target,
        weight,
        pred_decode=pred_decode,
        targets_decode=targets_decode)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = KFLoss()(
        pred,
        target,
        weight,
        pred_decode=pred_decode,
        targets_decode=targets_decode,
        reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = KFLoss()(
        pred,
        target,
        weight,
        pred_decode=pred_decode,
        targets_decode=targets_decode,
        avg_factor=10)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.skipif(
    digit_version(mmcv.__version__) <= digit_version('1.5.0'),
    reason='requires mmcv>=1.5.0')
def test_rotated_iou_losses():
    """Tests convex regression losses."""
    pred = torch.rand((10, 5)).cuda()
    target = torch.rand((10, 5)).cuda()
    weight = torch.rand((10, )).cuda()

    # Test loss mode
    loss = RotatedIoULoss(linear=True)(pred, target)
    assert isinstance(loss, torch.Tensor)

    loss = RotatedIoULoss(mode='linear')(pred, target)
    assert isinstance(loss, torch.Tensor)

    loss = RotatedIoULoss(mode='log')(pred, target)
    assert isinstance(loss, torch.Tensor)

    loss = RotatedIoULoss(mode='square')(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward
    loss = RotatedIoULoss()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with weight
    loss = RotatedIoULoss()(pred, target, weight)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = RotatedIoULoss()(pred, target, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = RotatedIoULoss()(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)
