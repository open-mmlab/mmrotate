# Copyright (c) OpenMMLab. All rights reserved.

import mmcv
import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply

from .rotated_anchor_head import RotatedAnchorHead
from ..builder import ROTATED_HEADS
from ... import multiclass_nms_rotated
from .utils import covariance_output_to_cholesky, compute_mean_covariance_torch


@ROTATED_HEADS.register_module()
class ProbabilisticRetinaNetHead(RotatedAnchorHead):
    """
        The head used in ProbabilisticRetinaNet for object class probability estimation, box regression, box covariance estimation.
        It has three subnets for the three tasks, with a common structure but separate parameters.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 use_dropout=None,
                 dropout_rate=None,
                 compute_cls_var=None,
                 compute_bbox_cov=None,
                 bbox_cov_dims=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(ProbabilisticRetinaNetHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        num_convs = 4  # 4
        prior_prob = 0.01  # 0.01
        self.compute_cls_var = compute_cls_var
        self.compute_bbox_cov = compute_bbox_cov
        self.bbox_cov_dims = bbox_cov_dims
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Conv2d(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1, ))
            self.cls_convs.append(self.relu)
            self.reg_convs.append(
                nn.Conv2d(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1, ))
            self.reg_convs.append(self.relu)

            if self.use_dropout:
                self.cls_convs.append(nn.Dropout(p=self.dropout_rate))
                self.reg_convs.append(nn.Dropout(p=self.dropout_rate))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

        # for modules in [
        #     self.cls_convs,
        #     self.reg_convs,
        #     self.retina_cls,
        #     self.retina_reg
        # ]:
        #     for layer in modules.modules():
        #         if isinstance(layer, nn.Conv2d):
        #             nn.init.normal_(layer.weight, mean=0, std=0.01)
        #             nn.init.constant_(layer.bias, 0)
        #
        # # Use prior in model initialization to improve stability
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # nn.init.constant_(self.retina_cls.bias, bias_value)

        # Create subnet for classification variance estimation.
        if self.compute_cls_var:
            self.cls_var = nn.Conv2d(
                self.feat_channels,
                self.num_anchors * self.cls_out_channels,
                kernel_size=3,
                stride=1,
                padding=1)

            # for layer in self.cls_var.modules():
            #     if isinstance(layer, nn.Conv2d):
            #         torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
            #         torch.nn.init.constant_(layer.bias, -10.0)

        # Create subnet for bounding box covariance estimation.
        if self.compute_bbox_cov:
            self.bbox_cov = nn.Conv2d(
                self.feat_channels,
                self.num_anchors * self.bbox_cov_dims,
                kernel_size=3,
                stride=1,
                padding=1)

            # for layer in self.bbox_cov.modules():
            #     if isinstance(layer, nn.Conv2d):
            #         torch.nn.init.normal_(layer.weight, mean=0, std=0.0001)
            #         torch.nn.init.constant_(layer.bias, 0)

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
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        logits = self.retina_cls(cls_feat)
        bbox_reg = self.retina_reg(reg_feat)
        logits_var = self.cls_var(cls_feat) if self.compute_cls_var else None
        bbox_cov = self.bbox_cov(reg_feat) if self.compute_bbox_cov else None

        return logits, bbox_reg, logits_var, bbox_cov

    def loss_single(self, cls_score, bbox_pred, logits_var, bbox_conv, anchors, labels, label_weights,
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
        logits_var = logits_var.permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, logits_var, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        bbox_conv = bbox_conv.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_conv,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'logits_var', 'bbox_conv'))
    def loss(self,
             cls_scores,
             bbox_preds,
             logits_var,
             bbox_conv,
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
            logits_var,
            bbox_conv,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           cls_vars_list,
                           bbox_covs_list,
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
                scale level with shape (num_anchors * 5, H, W).
            angle_cls_list (list[Tensor]): Angle deltas for a single
                scale level with shape (num_anchors * coding_len, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 5).
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
        mlvl_bboxes_cov = []
        mlvl_scores_var = []
        for cls_score, bbox_pred, cls_var, bbox_cov, anchors in zip(
                cls_score_list, bbox_pred_list, cls_vars_list, bbox_covs_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            # todo 计算分类分数时为什么需要在得到的分布中进行重采样？ 仔细阅读bayesOd论文
            cls_var = cls_var.permute(1, 2,
                                      0).reshape(-1, self.cls_out_channels)
            cls_dists = torch.distributions.normal.Normal(
                cls_score, scale=torch.sqrt(torch.exp(cls_var))
            )
            cls_score = cls_dists.rsample((10,))
            cls_score = torch.mean(cls_score.sigmoid(), 0)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            bbox_cov = bbox_cov.permute(1, 2, 0).reshape(-1, 5)

            nms_pre = cfg.get('nms_pre', -1)
            if scores.shape[0] > nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # Remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                bbox_cov = bbox_cov[topk_inds, :]
                cls_var = cls_var[topk_inds, :]
            bbox_cov = torch.clamp(bbox_cov, -7, 7)
            cholesky_decomp = covariance_output_to_cholesky(bbox_cov, num_factor=5)

            multivariate_normal_samples = torch.distributions.MultivariateNormal(
                bbox_pred, scale_tril=cholesky_decomp)

            # Define monte-carlo samples
            distributions_samples = multivariate_normal_samples.rsample(
                (1000,))
            samples_anchors = torch.repeat_interleave(
                anchors.unsqueeze(0), 1000, dim=0)
            t_dist_samples = self.bbox_coder.decode(
                samples_anchors, distributions_samples, max_shape=img_shape)
            # todo 函数维度修改
            t_dist_samples = torch.transpose(
                torch.transpose(t_dist_samples, 0, 1), 1, 2
            )
            bboxes, bboxes_cov = compute_mean_covariance_torch(t_dist_samples)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_bboxes_cov.append(bboxes_cov)
            mlvl_scores_var.append(cls_var)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # Angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_bboxes_cov = torch.cat(mlvl_bboxes_cov)
        mlvl_scores_var = torch.cat(mlvl_scores_var)
        # todo 使用标准nms，对分数不做改动了？
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # Remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels, keep = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img, return_inds=True)
            det_bboxes_cov = mlvl_bboxes_cov[keep]
            # todo bboxes shape as N*5*5, how to do with it? transfer it to (N,)?
            return torch.cat([det_bboxes, det_bboxes_cov[:, None]], 1), det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    # todo 认知不确定性暂未解决
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_clses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   cls_vars,
                   bbox_covs,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_vars:
            bbox_covs:
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
            >>> # Note the input lists are over different levels, not images
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
            cls_vars_list = [
                cls_vars[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_covs_list = [
                bbox_covs[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # Some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    cls_vars_list,
                                                    bbox_covs_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    cls_vars_list,
                                                    bbox_covs_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list