# Copyright (c) OpenMMLab. All rights reserved.
from inspect import signature

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmrotate.core import (aug_multiclass_nms_rotated, bbox_mapping_back,
                           build_assigner, build_bbox_coder,
                           build_prior_generator, build_sampler,
                           multiclass_nms_rotated, obb2hbb,
                           rotated_anchor_inside_flags)
from ..builder import ROTATED_HEADS, build_loss


@ROTATED_HEADS.register_module()
class RotatedAnchorHead(BaseDenseHead):
    """Rotated Anchor-based head (RotatedRPN, RotatedRetinaNet, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied on decoded bounding boxes. Default: False
        assign_by_circumhbbox (str): If None, assigner will assign according to
            the IoU between anchor and GT (OBB), called RetinaNet-OBB.
            If angle definition method, assigner will assign according to the
            IoU between anchor and GT's circumbox (HBB), called RetinaNet-HBB.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='RotatedAnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[1.0, 0.5, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 assign_by_circumhbbox='oc',
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(RotatedAnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # TODO better way to determine whether sample or not
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox
        self.assign_by_circumhbbox = assign_by_circumhbbox
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.anchor_generator = build_prior_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 5, 1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 5.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple (list[Tensor]):

                - anchor_list (list[Tensor]): Anchors of each image.
                - valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if self.assign_by_circumhbbox is not None:
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, gt_bboxes_assign, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                anchors, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
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

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 5).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        result_list = []
        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def aug_test(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 6),
                where 6 represent (x, y, w, h, a, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            bbox_outputs = self.get_bboxes(
                *outs,
                img_metas=img_meta,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=False)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)

        merged_scores, merged_labels = torch.max(merged_scores[:, :-1], dim=1)
        merged_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
        if merged_bboxes.numel() == 0:
            return [
                (merged_bboxes, merged_labels),
            ]

        det_bboxes, det_labels = aug_multiclass_nms_rotated(
            merged_bboxes, merged_labels, self.test_cfg.score_thr,
            self.test_cfg.nms, self.test_cfg.max_per_img, self.num_classes)

        if rescale:
            # angle should not be rescaled
            merged_bboxes[:, :4] *= merged_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (det_bboxes, det_labels),
        ]

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores
