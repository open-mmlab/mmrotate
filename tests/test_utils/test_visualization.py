# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import numpy as np

from mmrotate.core import visualization as vis
from mmrotate.datasets import DOTADataset, HRSCDataset, SARDataset


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


def test_palette():
    # test list
    palette = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    palette_ = vis.get_palette(palette, 3)
    for color, color_ in zip(palette, palette_):
        assert color == color_

    # test tuple
    palette = vis.get_palette((1, 2, 3), 3)
    assert len(palette) == 3
    for color in palette:
        assert color == (1, 2, 3)

    # test color str
    palette = vis.get_palette('red', 3)
    assert len(palette) == 3
    for color in palette:
        assert color == (255, 0, 0)

    # test dataset str
    palette = vis.get_palette('dota', len(DOTADataset.CLASSES))
    assert len(palette) == len(DOTADataset.CLASSES)

    palette = vis.get_palette('sar', len(SARDataset.CLASSES))
    assert len(palette) == len(SARDataset.CLASSES)

    palette = vis.get_palette('hrsc', len(HRSCDataset.HRSC_CLASS))
    assert len(palette) == len(HRSCDataset.HRSC_CLASS)

    palette = vis.get_palette('hrsc_classwise', len(HRSCDataset.HRSC_CLASSES))
    assert len(palette) == len(HRSCDataset.HRSC_CLASSES)

    # test random
    palette1 = vis.get_palette('random', 3)
    palette2 = vis.get_palette(None, 3)
    for color1, color2 in zip(palette1, palette2):
        assert isinstance(color1, tuple)
        assert isinstance(color2, tuple)
        assert color1 == color2
