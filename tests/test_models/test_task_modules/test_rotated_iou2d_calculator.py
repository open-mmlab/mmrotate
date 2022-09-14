# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.testing import assert_allclose

from mmrotate.core.bbox.iou_calculators import (FakeRBboxOverlaps2D,
                                                RBboxOverlaps2D,
                                                rbbox_overlaps)
from mmrotate.core.bbox.structures import RotatedBoxes


class TestRBoxOverlaps2D(TestCase):

    def _construct_rbbox(self, num_bbox=None):
        h = int(np.random.randint(3, 1000))
        w = int(np.random.randint(3, 1000))
        if num_bbox is None:
            num_bbox = np.random.randint(1, 10)
        cx, cy, bw, bh, angle = torch.rand(num_bbox, 5).T
        bboxes = torch.stack([cx * w, cy * h, w * bw, h * bh, angle], dim=-1)
        return bboxes, num_bbox

    def test_rbbox_overlaps_2d(self):
        overlap = RBboxOverlaps2D()
        bboxes1, num_bbox = self._construct_rbbox()
        bboxes2, _ = self._construct_rbbox(num_bbox)
        ious = overlap(bboxes1, bboxes2, 'iou', True)
        self.assertEqual(ious.size(), (num_bbox, ), ious.size())
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))

        # use RotatedBoxes
        bboxes1 = RotatedBoxes(bboxes1)
        bboxes2 = RotatedBoxes(bboxes2)
        ious = overlap(bboxes1, bboxes2, 'iou', True)
        self.assertEqual(ious.size(), (num_bbox, ), ious.size())
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))

        # is_aligned is True, bboxes.size(-1) == 6 (include score)
        overlap = RBboxOverlaps2D()
        bboxes1, num_bbox = self._construct_rbbox()
        bboxes2, _ = self._construct_rbbox(num_bbox)
        bboxes1 = torch.cat((bboxes1, torch.rand((num_bbox, 1))), dim=-1)
        bboxes2 = torch.cat((bboxes2, torch.rand((num_bbox, 1))), dim=-1)
        ious = overlap(bboxes1, bboxes2, 'iou', True)
        self.assertEqual(ious.size(), (num_bbox, ), ious.size())
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))

        # is_aligned is True, bboxes1.size(-2) == 0
        bboxes1 = torch.empty((0, 5))
        bboxes2 = torch.empty((0, 5))
        ious = overlap(bboxes1, bboxes2, 'iou', True)
        self.assertEqual(ious.size(), torch.Size([0, 1]))
        self.assertTrue(torch.all(ious == torch.empty((0, ))))
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))

        # is_aligned is False
        bboxes1, num_bbox1 = self._construct_rbbox()
        bboxes2, num_bbox2 = self._construct_rbbox()
        ious = overlap(bboxes1, bboxes2, 'iou')
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))
        self.assertEqual(ious.size(), (num_bbox1, num_bbox2))

    def test_rbbox_overlaps(self):
        # test allclose between rbbox_overlaps and the original official
        # implementation.
        bboxes1 = torch.FloatTensor([[1.0, 1.0, 3.0, 4.0, 0.5],
                                     [2.0, 2.0, 3.0, 4.0, 0.6],
                                     [7.0, 7.0, 8.0, 8.0, 0.4]])
        bboxes2 = torch.FloatTensor([
            [0.0, 2.0, 2.0, 5.0, 0.3],
            [2.0, 1.0, 3.0, 3.0, 0.5],
            [5.0, 5.0, 6.0, 7.0, 0.4],
        ])
        ious = rbbox_overlaps(bboxes1, bboxes2, 'iou', is_aligned=True)
        # the gt is got with four decimal precision.
        expected_ious = torch.FloatTensor([0.3708, 0.4487, 0.3622])
        assert_allclose(ious, expected_ious)

        # test mode 'iof'
        ious = rbbox_overlaps(bboxes1, bboxes2, 'iof', is_aligned=True)
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))
        self.assertEqual(ious.size(), (bboxes1.size(0), ))
        ious = rbbox_overlaps(bboxes1, bboxes2, 'iof')
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))
        self.assertEqual(ious.size(), (bboxes1.size(0), bboxes2.size(0)))

    def test_fake_rbbox_overlaps_2d(self):
        overlap = FakeRBboxOverlaps2D()
        bboxes1, num_bbox = self._construct_rbbox()
        bboxes2, _ = self._construct_rbbox(num_bbox)
        ious = overlap(bboxes1, bboxes2, 'iou', True)
        self.assertEqual(ious.size(), (num_bbox, ), ious.size())
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))

        # use RotatedBoxes
        bboxes1 = RotatedBoxes(bboxes1)
        bboxes2 = RotatedBoxes(bboxes2)
        ious = overlap(bboxes1, bboxes2, 'iou', True)
        self.assertEqual(ious.size(), (num_bbox, ), ious.size())
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))

        # is_aligned is True, bboxes.size(-1) == 6 (include score)
        overlap = RBboxOverlaps2D()
        bboxes1, num_bbox = self._construct_rbbox()
        bboxes2, _ = self._construct_rbbox(num_bbox)
        bboxes1 = torch.cat((bboxes1, torch.rand((num_bbox, 1))), dim=-1)
        bboxes2 = torch.cat((bboxes2, torch.rand((num_bbox, 1))), dim=-1)
        ious = overlap(bboxes1, bboxes2, 'iou', True)
        self.assertEqual(ious.size(), (num_bbox, ), ious.size())
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))

        # is_aligned is True, bboxes1.size(-2) == 0
        bboxes1 = torch.empty((0, 5))
        bboxes2 = torch.empty((0, 5))
        ious = overlap(bboxes1, bboxes2, 'iou', True)
        self.assertEqual(ious.size(), torch.Size([0, 1]))
        self.assertTrue(torch.all(ious == torch.empty((0, ))))
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))

        # is_aligned is False
        bboxes1, num_bbox1 = self._construct_rbbox()
        bboxes2, num_bbox2 = self._construct_rbbox()
        ious = overlap(bboxes1, bboxes2, 'iou')
        self.assertTrue(torch.all(ious >= -1) and torch.all(ious <= 1))
        self.assertEqual(ious.size(), (num_bbox1, num_bbox2))
