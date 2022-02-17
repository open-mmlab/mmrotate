# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import numpy as np

from mmrotate.core import visualization as vis


def test_imshow_det_bboxes():
    """Test imshow det bboxes."""
    tmp_filename = 'tests/det_bboxes_image/image.jpg'
    image = np.ones((10, 10, 3), np.uint8)
    ori_image = image.copy()
    bbox = np.array([[2.5, 2, 2.5, 2, 0, 0.9], [4.5, 5, 2.5, 2, 0, 0.9]])
    label = np.array([0, 1])
    out_image = vis.imshow_det_rbboxes(
        image, bbox, label, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert image.shape == out_image.shape
    assert not np.allclose(ori_image, out_image)
    os.remove(tmp_filename)

    # test grayscale images
    image = np.ones((10, 10), np.uint8)
    ori_image = image.copy()
    bbox = np.array([[2.5, 2, 2.5, 2, 0, 0.9], [4.5, 5, 2.5, 2, 0, 0.9]])
    label = np.array([0, 1])
    out_image = vis.imshow_det_rbboxes(
        image, bbox, label, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert ori_image.shape == out_image.shape[:2]
    os.remove(tmp_filename)

    # test shaped (0,)
    image = np.ones((10, 10, 3), np.uint8)
    bbox = np.ones((0, 6))
    label = np.ones((0, ))
    vis.imshow_det_rbboxes(
        image, bbox, label, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    os.remove(tmp_filename)
