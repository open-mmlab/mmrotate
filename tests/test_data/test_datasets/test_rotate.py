# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmdet.datasets import build_dataset

from mmrotate.datasets.sar import SARDataset


def _create_dummy_results():
    """Create dummy results."""
    boxes = [
        np.array([[4.3150e+02, 7.0600e+02, 6.7686e+01, 2.1990e+01, 2.9842e-02],
                  [5.6351e+02, 5.3575e+02, 1.0018e+02, 1.8971e+01, 5.5499e-02],
                  [5.7450e+02, 5.8450e+02, 9.5567e+01, 2.1094e+01,
                   8.4012e-02]])
    ]
    return [boxes]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('angle_version', ['oc'])
def test_dota_dataset(angle_version):
    """Test DOTA dataset.

    Args:
        angle_version (str, optional): Angle representations.
    """
    # test CLASSES
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    data_config = dict(
        type=SARDataset,
        classes=['plane'],
        version=angle_version,
        ann_file='tests/data/labelTxt/',
        img_prefix='tests/data/images/',
        pipeline=train_pipeline)
    dataset = build_dataset(data_config)
    assert dataset.CLASSES == ['plane']

    # test eval
    fake_results = _create_dummy_results()
    eval_results = dataset.evaluate(fake_results)
    assert eval_results['mAP'] > 0
