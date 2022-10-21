# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmcv.ops import batched_nms
from mmdet.models.dense_heads import RPNHead
from mmdet.structures.bbox import (BaseBoxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox2hbox


@MODELS.register_module()
class OrientedRPNHead(RPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method, which use horizontal bboxes for NMS,
        but return the rotated bboxes result.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            hbboxes = rbox2hbox(bboxes)
            det_bboxes, keep_idxs = batched_nms(hbboxes, results.scores,
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            if isinstance(results.bboxes, BaseBoxes):
                results_.bboxes = results.bboxes.empty_boxes()
            else:
                results_.bboxes = results.scores.new_zeros(0, 4)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results
