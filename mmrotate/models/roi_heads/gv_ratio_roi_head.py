# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import bbox2roi

from ..builder import ROTATED_HEADS
from .rotate_standard_roi_head import RotatedStandardRoIHead


@ROTATED_HEADS.register_module()
class GVRatioRoIHead(RotatedStandardRoIHead):
    """Gliding vertex roi head including one bbox head."""

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                bbox_results['fix_pred'],
                bbox_results['ratio_pred'],
            )
        return outs

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, fix_pred, ratio_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            fix_pred=fix_pred,
            ratio_pred=ratio_pred,
            bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        bbox_results['fix_pred'],
                                        bbox_results['ratio_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        fix_pred = bbox_results['fix_pred'],
        ratio_pred = bbox_results['ratio_pred'],
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            fix_pred = fix_pred[0].split(num_proposals_per_img, 0)
            ratio_pred = ratio_pred[0].split(num_proposals_per_img, 0)

        else:
            bbox_pred = (None, ) * len(proposals)
            fix_pred = (None, ) * len(proposals)
            ratio_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                fix_pred[i],
                ratio_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=self.test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels
