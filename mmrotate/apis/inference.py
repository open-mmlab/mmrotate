# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from mmrotate.core import get_multiscale_patch, merge_results, slide_window


def inference_detector_by_patches(model,
                                  img,
                                  sizes,
                                  steps,
                                  ratios,
                                  merge_iou_thr,
                                  bs=1):
    """inference patches with the detector.

    Split huge image(s) into patches and inference them with the detector.
    Finally, merge patch results on one huge image by nms.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray or): Either an image file or loaded image.
        sizes (list): The sizes of patches.
        steps (list): The steps between two patches.
        ratios (list): Image resizing ratios for multi-scale detecting.
        merge_iou_thr (float): IoU threshold for merging results.
        bs (int): Batch size, must greater than or equal to 1.

    Returns:
        list[np.ndarray]: Detection results.
    """
    assert bs >= 1, 'The batch size must greater than or equal to 1'
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadPatchFromImage'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    if not isinstance(img, np.ndarray):
        img = mmcv.imread(img)
    height, width = img.shape[:2]
    sizes, steps = get_multiscale_patch(sizes, steps, ratios)
    windows = slide_window(width, height, sizes, steps)

    results = []
    start = 0
    while True:
        # prepare patch data
        patch_datas = []
        if (start + bs) > len(windows):
            end = len(windows)
        else:
            end = start + bs
        for window in windows[start:end]:
            data = dict(img=img, win=window.tolist())
            data = test_pipeline(data)
            patch_datas.append(data)
        data = collate(patch_datas, samples_per_gpu=len(patch_datas))
        # just get the actual data from DataContainer
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            results.extend(model(return_loss=False, rescale=True, **data))

        if end >= len(windows):
            break
        start += bs

    results = merge_results(
        results,
        windows[:, :2],
        img_shape=(width, height),
        iou_thr=merge_iou_thr,
        device=device)
    return results
