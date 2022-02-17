# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES

from .utils import _check_fields


def check_result_same(results, pipeline_results):
    """Check whether the `pipeline_results` is the same with the predefined
    `results`.

    Args:
        results (dict): Predefined results which should be the standard output
            of the transform pipeline.
        pipeline_results (dict): Results processed by the transform pipeline.
    """
    # check image
    _check_fields(results, pipeline_results,
                  results.get('img_fields', ['img']))
    # check bboxes
    _check_fields(results, pipeline_results, results.get('bbox_fields', []))
    # check gt_labels
    if 'gt_labels' in results:
        assert np.equal(results['gt_labels'],
                        pipeline_results['gt_labels']).all()


def construct_toy_data():
    """Construct toy data."""
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['img'] = img
    results['img_shape'] = img.shape
    results['img_fields'] = ['img']
    # bboxes
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    results['gt_bboxes'] = np.array([[4., 4., 2., 1., 0.]], dtype=np.float32)
    results['gt_bboxes_ignore'] = np.array([[2., 4., 3., 1., 0.]],
                                           dtype=np.float32)
    # labels
    results['gt_labels'] = np.array([1], dtype=np.int64)

    return results


def test_rresize():
    """Test resize for rbboxes."""
    results = construct_toy_data()
    transform = dict(
        type='RResize',
        img_scale=(8, 8),
    )
    rresize_module = build_from_cfg(transform, PIPELINES)
    results_rresize = rresize_module(copy.deepcopy(results))
    assert results_rresize['img_shape'] == (4, 8, 3)


def test_rflip():
    """Test flip for rbboxes."""
    results = construct_toy_data()

    # test horizontal flip
    transform = dict(
        type='RRandomFlip',
        flip_ratio=1.0,
        direction='horizontal',
        version='oc',
    )
    rflip_module = build_from_cfg(transform, PIPELINES)
    results_rflip = rflip_module(copy.deepcopy(results))
    assert np.allclose(results_rflip['gt_bboxes'][0][-1], 1.57, atol=1.e-2)

    # test vertical flip
    transform = dict(
        type='RRandomFlip',
        flip_ratio=1.0,
        direction='vertical',
        version='le135',
    )
    rflip_module = build_from_cfg(transform, PIPELINES)
    results_rflip = rflip_module(copy.deepcopy(results))
    assert np.allclose(results_rflip['gt_bboxes'][0][-1], 0, atol=1.e-2)

    # test diagonal flip
    transform = dict(
        type='RRandomFlip',
        flip_ratio=1.0,
        direction='diagonal',
        version='le90',
    )
    rflip_module = build_from_cfg(transform, PIPELINES)
    results_rflip = rflip_module(copy.deepcopy(results))
    assert np.allclose(results_rflip['gt_bboxes'][0][-1], 0, atol=1.e-2)


def test_rotate():
    """Test rotation for rbboxes."""
    results = construct_toy_data()

    # test PolyRandomRotate
    transform = dict(
        type='PolyRandomRotate',
        rotate_ratio=1.0,
        angles_range=180,
        auto_bound=False,
        version='oc')
    rotate_module = build_from_cfg(transform, PIPELINES)
    rotate_module(copy.deepcopy(results))
