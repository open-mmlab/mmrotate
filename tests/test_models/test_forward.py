# Copyright (c) OpenMMLab. All rights reserved.
"""pytest tests/test_forward.py."""
import copy
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def _replace_r50_with_r18(model):
    """Replace ResNet50 with ResNet18 in config."""
    model = copy.deepcopy(model)
    if model.backbone.type == 'ResNet':
        model.backbone.depth = 18
        model.backbone.base_channels = 2
        model.neck.in_channels = [2, 4, 8, 16]
    return model


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('cfg_file', [
    'rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_oc.py',
    'rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py',
    'rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_le135.py',
    'rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_le90.py',
    'r3det/r3det_r50_fpn_1x_dota_oc.py',
    'r3det/r3det_refine_r50_fpn_1x_dota_oc.py',
    's2anet/s2anet_r50_fpn_1x_dota_le135.py',
    'rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py',
    'kld/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc.py',
    'kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc.py',
    'kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc.py',
    'kfiou/r3det_refine_kfiou_ln_r50_fpn_1x_dota_oc.py',
    'kfiou/s2anet_kfiou_ln_r50_fpn_1x_dota_le135.py',
    'gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc.py',
    'cfa/cfa_r50_fpn_1x_dota_oc.py',
])
def test_single_stage_forward_gpu(cfg_file):
    """Test single stage forward (GPU).

    Args:
        cfg_file (str): config of single stage detector.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')

    model = _get_detector_cfg(cfg_file)
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (2, 3, 128, 128)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    detector = detector.cuda()
    gt_bboxes = [b for b in mm_inputs['gt_bboxes']]
    gt_labels = [g for g in mm_inputs['gt_labels']]
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('cfg_file', [
    'gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90.py',
    'roi_trans/roi_trans_r50_fpn_1x_dota_oc.py',
    'roi_trans/roi_trans_r50_fpn_1x_dota_le135.py',
    'roi_trans/roi_trans_r50_fpn_1x_dota_le90.py',
    'rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py',
    'oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
])
def test_two_stage_forward_gpu(cfg_file):
    """Test two stage forward (GPU).

    Args:
        cfg_file (str): config of two stage detector.
    """
    model = _get_detector_cfg(cfg_file)
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)
    detector = detector.cuda()

    input_shape = (1, 3, 128, 128)

    # Test forward train with a non-empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    losses = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    losses = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()


def _demo_mm_inputs(input_shape=(1, 3, 300, 300),
                    num_items=None, num_classes=10,
                    with_semantic=False):  # yapf: disable
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': np.array([1.1, 1.2, 1.1, 1.2]),
        'flip': False,
        'flip_direction': None,
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        x_ctr = cx * W
        y_ctr = cy * H
        w = (W * bw).clip(x_ctr / 2, W - x_ctr / 2)
        h = (H * bh).clip(y_ctr / 2, H - y_ctr / 2)

        boxes = np.vstack([x_ctr, y_ctr, w, h, np.zeros(num_boxes).T]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes).cuda())
        gt_labels.append(torch.LongTensor(class_idxs).cuda())

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True).cuda(),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
        'gt_masks': gt_masks,
    }

    if with_semantic:
        # assume gt_semantic_seg using scale 1/8 of the img
        gt_semantic_seg = np.random.randint(
            0, num_classes, (1, 1, H // 8, W // 8), dtype=np.uint8)
        mm_inputs.update(
            {'gt_semantic_seg': torch.ByteTensor(gt_semantic_seg)})

    return mm_inputs
