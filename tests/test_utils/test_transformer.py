# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

import mmrotate.core.bbox.transforms as rtf


def test_transforms():
    """Test transforms."""

    # test obb2poly_np
    rbboxes = np.array([[5., 3., 3., 2., 0., 0.95],
                        [5., 3., 3., 2., 1.2, 0.95]])
    polys1 = rtf.obb2poly_np(rbboxes, 'oc')
    polys2 = rtf.obb2poly_np(rbboxes, 'le135')
    polys3 = rtf.obb2poly_np(rbboxes, 'le90')
    assert np.allclose(polys1, polys2)
    assert np.allclose(polys2, polys3)

    # test poly2obb_le135
    polys = torch.from_numpy(polys2[:, :-1])
    assert np.allclose(rtf.poly2obb_le135(polys), rbboxes[:, :-1])

    # test hbb2obb
    hbboxes = torch.tensor([[0., 0., 4., 4.], [1., 3., 4., 4.]])
    obboxes1 = rtf.hbb2obb(hbboxes, 'oc')
    obboxes2 = rtf.hbb2obb(hbboxes, 'le135')
    obboxes3 = rtf.hbb2obb(hbboxes, 'le90')
    assert not np.allclose(obboxes1.numpy(), obboxes2)
    assert np.allclose(obboxes2.numpy(), obboxes3)
