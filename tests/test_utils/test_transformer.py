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

    # test full360
    # Check obb2poly and poly2obb is inverse function in full360 rotation
    for angle in np.linspace(-.9 * np.pi, .9 * np.pi, 4):
        # numpy version
        box_np = np.array((100, 100, 80, 50, angle), dtype=np.float32)
        pts_np = rtf.obb2poly_np(box_np[None], version='full360')[0]
        box2_np = rtf.poly2obb_np(pts_np, version='full360')
        np.testing.assert_almost_equal(box_np, box2_np, decimal=4)

        # torch version
        box_torch = torch.tensor((100, 100, 80, 50, angle),
                                 dtype=torch.float32)
        pts_torch = rtf.obb2poly(box_torch[None], version='full360')[0]
        box2_torch = rtf.poly2obb(pts_torch, version='full360')[0]
        torch.testing.assert_close(box_torch, box2_torch, rtol=1e-4, atol=1e-4)

        # compatibility
        torch.testing.assert_close(
            box_torch, torch.from_numpy(box_np), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            pts_torch, torch.from_numpy(pts_np), rtol=1e-4, atol=1e-4)
