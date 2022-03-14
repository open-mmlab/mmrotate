# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.ops import nms_rotated


def merge_results(results, offsets, iou_thr=0.1, device='cpu'):
    """Merge patch results via nms.

    Args:
        results (list[np.ndarray]): A list of patches results.
        offsets (np.ndarray): Positions of the left top points of patches.
        iou_thr (float): The IoU threshold of NMS.
        device (str): The device to call nms.

    Retunrns:
        list[np.ndarray]: Detection results after merging.
    """
    assert len(results) == offsets.shape[0], 'The `results` should has the ' \
                                             'same length with `offsets`.'
    merged_results = []
    for results_pre_cls in zip(*results):
        tran_dets = []
        for dets, offset in zip(results_pre_cls, offsets):
            dets[:, :2] += offset
            tran_dets.append(dets)
        tran_dets = np.concatenate(tran_dets, axis=0)

        if tran_dets.size == 0:
            merged_results.append(tran_dets)
        else:
            tran_dets = torch.from_numpy(tran_dets)
            tran_dets = tran_dets.to(device)
            nms_dets, _ = nms_rotated(tran_dets[:, :5], tran_dets[:, -1],
                                      iou_thr)
            merged_results.append(nms_dets.cpu().numpy())
    return merged_results
