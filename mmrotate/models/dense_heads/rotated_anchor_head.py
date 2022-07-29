# Copyright (c) OpenMMLab. All rights reserved.
from inspect import signature
from typing import Optional

import torch
from mmcv.ops import batched_nms
from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.models.utils import unmap
from mmengine.config import ConfigDict
from mmengine.data import InstanceData
from torch import Tensor

from mmrotate.models.task_modules import rotated_anchor_inside_flags
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import obb2hbb
from ..test_time_augs import merge_aug_results


@MODELS.register_module()
class RotatedAnchorHead(AnchorHead):
    """Rotated Anchor-based head.

    Args:
        assign_by_hbb (str, optional): If None, assigner will assign according
            to the IoU between anchor and GT (OBB), called RetinaNet-OBB.
            If angle definition method, assigner will assign according to the
            IoU between anchor and GT's circumbox (HBB), called RetinaNet-HBB.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 assign_by_hbb: Optional[str] = None,
                 **kwargs) -> None:
        self.assign_by_hbb = assign_by_hbb
        super().__init__(num_classes, in_channels, **kwargs)

    def _get_targets_single(self,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.
        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.
        Returns:
            tuple:
                - labels (Tensor): Labels of each level.
                - label_weights (Tensor): Label weights of each level.
                - bbox_targets (Tensor): BBox targets of each level.
                - bbox_weights (Tensor): BBox weights of each level.
                - pos_inds (Tensor): positive samples indexes.
                - neg_inds (Tensor): negative samples indexes.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if self.assign_by_hbb is not None:
            gt_instances.bboxes = obb2hbb(gt_instances.bboxes,
                                          self.assign_by_hbb)

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    # TODO remove this in boxlist
    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.
        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 5).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 5).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 5).
            avg_factor (int): Average factor that is used to average the loss.
        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls, loss_bbox

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.
        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.
        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
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

        if rescale:
            assert img_meta.get('scale_factor') is not None
            # angle should not be rescaled
            results.bboxes[:, :4] /= results.bboxes.new_tensor(
                img_meta['scale_factor'])

        if hasattr(results, 'score_factors'):
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size rotated bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w = results.bboxes[:, 2]
            h = results.bboxes[:, 3]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

        return results

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation.
        Args:
            aug_batch_feats (list[tuple[Tensor]]): The outer list
                indicates test-time augmentations and inner tuple
                indicate the multi-level feats from
                FPN, each Tensor should have a shape (B, C, H, W),
            aug_batch_img_metas (list[list[dict]]): Meta information
                of images under the different test-time augs
                (multiscale, flip, etc.). The outer list indicate
                the
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            with_ori_nms (bool): Whether execute the nms in original head.
                Defaults to False. It will be `True` when the head is
                adopted as `rpn_head`.
        Returns:
            list(obj:`InstanceData`): Detection results of the
            input images. Each item usually contains\
            following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances,).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # TODO: remove this for detr and deformdetr
        sig_of_get_results = signature(self.get_results)
        get_results_args = [
            p.name for p in sig_of_get_results.parameters.values()
        ]
        get_results_single_sig = signature(self._get_results_single)
        get_results_single_sig_args = [
            p.name for p in get_results_single_sig.parameters.values()
        ]
        assert ('with_nms' in get_results_args) and \
               ('with_nms' in get_results_single_sig_args), \
               f'{self.__class__.__name__}' \
               'does not support test-time augmentation '

        num_imgs = len(aug_batch_img_metas[0])
        aug_batch_results = []
        for x, img_metas in zip(aug_batch_feats, aug_batch_img_metas):
            outs = self.forward(x)
            batch_instance_results = self.get_results(
                *outs,
                img_metas=img_metas,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=with_ori_nms,
                **kwargs)
            aug_batch_results.append(batch_instance_results)

        # after merging, bboxes will be rescaled to the original image
        batch_results = merge_aug_results(aug_batch_results,
                                          aug_batch_img_metas)

        final_results = []
        for img_id in range(num_imgs):
            results = batch_results[img_id]
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels,
                                                self.test_cfg.nms)
            results = results[keep_idxs]
            # some nms operation may reweight the score such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:self.test_cfg.max_per_img]
            if rescale:
                # all results have been mapped to the original scale
                # in `merge_aug_results`, so just pass
                pass
            else:
                # map to the first aug image scale
                scale_factor = results.bboxes.new_tensor(
                    aug_batch_img_metas[0][img_id]['scale_factor'])
                results.bboxes = \
                    results.bboxes * scale_factor

            final_results.append(results)

        return final_results
