# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from parameterized import parameterized

from mmrotate.models.losses import (BCConvexGIoULoss, ConvexGIoULoss, GDLoss,
                                    GDLoss_v1, H2RBoxConsistencyLoss, KFLoss,
                                    RotatedIoULoss, SpatialBorderLoss)


class TestGDLoss(unittest.TestCase):

    def test_loss_with_reduction_override(self):
        pred = torch.rand((10, 5))
        target = torch.rand((10, 5)),
        weight = None

        with self.assertRaises(AssertionError):
            # only reduction_override from [None, 'none', 'mean', 'sum']
            # is not allowed
            reduction_override = True
            GDLoss('gwd')(
                pred, target, weight, reduction_override=reduction_override)

    @parameterized.expand([('gwd', (0, 5)), ('gwd', (10, 5)), ('kld', (10, 5)),
                           ('jd', (10, 5)), ('kld_symmax', (10, 5)),
                           ('kld_symmin', (10, 5))])
    def test_regression_losses(self, loss_type, input_shape):
        pred = torch.rand(input_shape)
        target = torch.rand(input_shape)
        weight = torch.rand(input_shape)

        # Test loss forward
        loss = GDLoss(loss_type)(pred, target)
        self.assertIsInstance(loss, torch.Tensor)

        # Test loss forward with weight
        loss = GDLoss(loss_type)(pred, target, weight)
        self.assertIsInstance(loss, torch.Tensor)

        # Test loss forward with reduction_override
        loss = GDLoss(loss_type)(pred, target, reduction_override='mean')
        self.assertIsInstance(loss, torch.Tensor)

        # Test loss forward with avg_factor
        loss = GDLoss(loss_type)(pred, target, avg_factor=10)
        self.assertIsInstance(loss, torch.Tensor)

        with self.assertRaises(ValueError):
            # loss can evaluate with avg_factor only if
            # reduction is None, 'none' or 'mean'.
            reduction_override = 'sum'
            GDLoss(loss_type)(
                pred,
                target,
                avg_factor=10,
                reduction_override=reduction_override)

        # Test loss forward with avg_factor and reduction
        for reduction_override in [None, 'none', 'mean']:
            GDLoss(loss_type)(
                pred,
                target,
                avg_factor=10,
                reduction_override=reduction_override)
            self.assertIsInstance(loss, torch.Tensor)


class TestGDLoss_v1(unittest.TestCase):

    def test_loss_with_reduction_override(self):
        pred = torch.rand((10, 5))
        target = torch.rand((10, 5)),
        weight = None

        with self.assertRaises(AssertionError):
            # only reduction_override from [None, 'none', 'mean', 'sum']
            # is not allowed
            reduction_override = True
            GDLoss_v1('gwd')(
                pred, target, weight, reduction_override=reduction_override)

    @parameterized.expand([('gwd', (0, 5)), ('gwd', (10, 5)), ('kld', (10, 5)),
                           ('bcd', (10, 5))])
    def test_regression_losses(self, loss_type, input_shape):
        pred = torch.rand(input_shape)
        target = torch.rand(input_shape)
        weight = torch.rand(input_shape)

        # Test loss forward
        loss = GDLoss_v1(loss_type)(pred, target)
        self.assertIsInstance(loss, torch.Tensor)

        # Test loss forward with weight
        loss = GDLoss_v1(loss_type)(pred, target, weight)
        self.assertIsInstance(loss, torch.Tensor)

        # Test loss forward with reduction_override
        loss = GDLoss_v1(loss_type)(pred, target, reduction_override='mean')
        self.assertIsInstance(loss, torch.Tensor)

        # Test loss forward with avg_factor
        loss = GDLoss_v1(loss_type)(pred, target, avg_factor=10)
        self.assertIsInstance(loss, torch.Tensor)

        # Test loss forward with avg_factor and reduction
        for reduction_override in [None, 'none', 'mean']:
            GDLoss_v1(loss_type)(
                pred, target, reduction_override=reduction_override)
            self.assertIsInstance(loss, torch.Tensor)


class TestConvexGIoULoss(unittest.TestCase):

    @parameterized.expand([(ConvexGIoULoss, ), (BCConvexGIoULoss, )])
    def test_loss_with_reduction_override(self, loss_class):
        pred = torch.rand((10, 18))
        target = torch.rand((10, 8)),
        weight = None

        with self.assertRaises(AssertionError):
            # only reduction_override from [None, 'none', 'mean', 'sum']
            # is not allowed
            reduction_override = True
            loss_class()(
                pred, target, weight, reduction_override=reduction_override)

    @parameterized.expand([(ConvexGIoULoss, ), (BCConvexGIoULoss, )])
    def test_regression_losses(self, loss_class):

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')

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


class TestSpatialBorderLoss(unittest.TestCase):

    def test_regression_losses(self):

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')

        pred = torch.rand((10, 18)).cuda()
        target = torch.rand((10, 8)).cuda()
        weight = torch.rand((10, )).cuda()

        # Test loss forward with weight
        loss = SpatialBorderLoss()(pred, target, weight)
        assert isinstance(loss, torch.Tensor)

        # Test loss forward with avg_factor
        loss = SpatialBorderLoss()(pred, target, weight, avg_factor=10)
        assert isinstance(loss, torch.Tensor)


class TestKFLoss(unittest.TestCase):

    def test_regression_losses(self):

        pred = torch.rand((10, 5))
        target = torch.rand((10, 5))
        weight = torch.rand((10, ))
        pred_decode = torch.rand((10, 5))
        targets_decode = torch.rand((10, 5))

        # Test loss forward
        loss = KFLoss()(
            pred,
            target,
            pred_decode=pred_decode,
            targets_decode=targets_decode)
        assert isinstance(loss, torch.Tensor)

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
            reduction_override='mean',
            pred_decode=pred_decode,
            targets_decode=targets_decode)
        assert isinstance(loss, torch.Tensor)

        # Test loss forward with avg_factor
        loss = KFLoss()(
            pred,
            target,
            weight,
            avg_factor=10,
            pred_decode=pred_decode,
            targets_decode=targets_decode)
        assert isinstance(loss, torch.Tensor)


class TestH2RBoxConsistencyLoss(unittest.TestCase):

    @parameterized.expand([(H2RBoxConsistencyLoss, ), (RotatedIoULoss, )])
    def test_loss_with_reduction_override(self, loss_class):
        pred = torch.rand((10, 18))
        target = torch.rand((10, 8)),
        weight = None

        with self.assertRaises(AssertionError):
            # only reduction_override from [None, 'none', 'mean', 'sum']
            # is not allowed
            reduction_override = True
            loss_class()(
                pred, target, weight, reduction_override=reduction_override)

    @parameterized.expand([(H2RBoxConsistencyLoss, ), (RotatedIoULoss, )])
    def test_regression_losses(self, loss_class):

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')

        pred = torch.rand((10, 5)).cuda()
        target = torch.rand((10, 5)).cuda()
        weight = torch.rand((10, )).cuda()

        # Test loss forward
        # loss = loss_class()(pred, target)
        # assert isinstance(loss, torch.Tensor)

        # Test loss forward with weight
        loss = loss_class()(pred, target, weight)
        assert isinstance(loss, torch.Tensor)

        # Test loss forward with reduction_override
        loss = loss_class()(pred, target, weight, reduction_override='mean')
        assert isinstance(loss, torch.Tensor)

        # Test loss forward with avg_factor
        loss = loss_class()(pred, target, weight, avg_factor=10)
        assert isinstance(loss, torch.Tensor)
