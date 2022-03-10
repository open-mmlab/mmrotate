# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrotate.core.bbox import rbbox_overlaps

predict = [[903.34, 1034.4, 1.81e-7, 1e-7, -0.312],
           [903.34, 1034.4, 1e-7, 1e-3, -0.312],
           [903.34, 1034.4, 1.81e7, 1e7, -0.312]]
gt = [[2.1525e+02, 7.5750e+01, 3.3204e+01, 1.2649e+01, 3.2175e-01],
      [3.0013e+02, 7.7144e+02, 4.9222e+02, 3.1368e+02, -1.3978e+00],
      [8.4887e+02, 6.9989e+02, 4.6854e+02, 3.0743e+02, -1.4008e+00],
      [8.5250e+02, 7.0250e+02, 7.6181e+02, 3.8200e+02, -1.3984e+00]]
expect_ious = [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]


def test_rbbox_overlaps_cpu():
    predict_tensor = torch.tensor(predict)
    gt_tensor = torch.tensor(gt)
    expect_ious_tensor = torch.tensor(expect_ious)
    ious = rbbox_overlaps(predict_tensor, gt_tensor)
    torch.allclose(ious, expect_ious_tensor, atol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_rbbox_overlaps_cuda():
    predict_tensor = torch.tensor(predict, device='cuda')
    gt_tensor = torch.tensor(gt, device='cuda')
    expect_ious_tensor = torch.tensor(expect_ious, device='cuda')
    ious = rbbox_overlaps(predict_tensor, gt_tensor)
    torch.allclose(ious, expect_ious_tensor, atol=1e-3)
