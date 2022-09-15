# Copyright (c) OpenMMLab. All rights reserved.
import unittest
import torch
from parameterized import parameterized
from mmrotate.models.losses import GDLoss, GDLoss_v1


class TestGDLoss(unittest.TestCase):

    def test_loss_with_reduction_override(self):
        pred = torch.rand((10, 5))
        target = torch.rand((10, 5)),
        weight = None

        with self.assertRaises(AssertionError):
            # only reduction_override from [None, 'none', 'mean', 'sum']
            # is not allowed
            reduction_override = True
            GDLoss('gwd')(pred, target, weight, reduction_override=reduction_override)


    @parameterized.expand([
        ('gwd', (0, 5)),
        ('gwd', (10, 5)),
        ('kld', (10, 5)),
        ('jd', (10, 5)),
        ('kld_symmax', (10, 5)),
        ('kld_symmin', (10, 5))
    ])
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
                pred, target, avg_factor=10, reduction_override=reduction_override)

        # Test loss forward with avg_factor and reduction
        for reduction_override in [None, 'none', 'mean']:
            GDLoss(loss_type)(
                pred, target, avg_factor=10, reduction_override=reduction_override)
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
            GDLoss_v1('gwd')(pred, target, weight, reduction_override=reduction_override)


    @parameterized.expand([
        ('gwd', (0, 5)),
        ('gwd', (10, 5)),
        ('kld', (10, 5)),
        ('bcd', (10, 5))
    ])
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
