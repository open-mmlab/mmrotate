# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d, chamfer_distance, min_area_polygons
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import select_single_mlvl
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmrotate.core import (build_assigner, build_sampler,
                           multiclass_nms_rotated, obb2poly, poly2obb)
from ..builder import ROTATED_HEADS, build_loss
from .utils import levels_to_images


def ChamferDistance2D(point_set_1,
                      point_set_2,
                      distance_weight=0.05,
                      eps=1e-12):
    """Compute the Chamfer distance between two point sets.

    Args:
        point_set_1 (torch.tensor): point set 1 with shape (N_pointsets,
                                    N_points, 2)
        point_set_2 (torch.tensor): point set 2 with shape (N_pointsets,
                                    N_points, 2)

    Returns:
        dist (torch.tensor): chamfer distance between two point sets
                             with shape (N_pointsets,)
    """
    assert point_set_1.dim() == point_set_2.dim()
    assert point_set_1.shape[-1] == point_set_2.shape[-1]
    assert point_set_1.dim() <= 3
    dist1, dist2, _, _ = chamfer_distance(point_set_1, point_set_2)
    dist1 = torch.sqrt(torch.clamp(dist1, eps))
    dist2 = torch.sqrt(torch.clamp(dist2, eps))
    dist = distance_weight * (dist1.mean(-1) + dist2.mean(-1)) / 2.0

    return dist


@ROTATED_HEADS.register_module()
class OrientedRepPointsHead(BaseDenseHead):
    """Oriented RepPoints head -<https://arxiv.org/pdf/2105.11111v4.pdf>. The
    head contains initial and refined stages based on RepPoints. The initial
    stage regresses coarse point sets, and the refine stage further regresses
    the fine point sets. The APAA scheme based on the quality of point set
    samples in the paper is employed in refined stage.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        feat_channels (int): Number of feature channels.
        point_feat_channels (int, optional): Number of channels of points
            features.
        stacked_convs (int, optional): Number of stacked convolutions.
        num_points (int, optional): Number of points in points set.
        gradient_mul (float, optional): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable, optional): points strides.
        point_base_scale (int, optional): Bbox scale for assigning labels.
        conv_bias (str, optional): The bias of convolution.
        loss_cls (dict, optional): Config of classification loss.
        loss_bbox_init (dict, optional): Config of initial points loss.
        loss_bbox_refine (dict, optional): Config of points loss in refinement.
        conv_cfg (dict, optional): The config of convolution.
        norm_cfg (dict, optional): The config of normlization.
        train_cfg (dict, optional): The config of train.
        test_cfg (dict, optional): The config of test.
        center_init (bool, optional): Whether to use center point assignment.
        top_ratio (float, optional): Ratio of top high-quality point sets.
                  Defaults to 0.4.
        init_qua_weight (float, optional): Quality weight of initial
                    stage.
        ori_qua_weight (float, optional): Orientation quality weight.
        poc_qua_weight (float, optional): Point-wise correlation
                    quality weight.
        version (str, optional): Angle representations. Defaults to 'oc'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_bias='auto',
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_spatial_init=dict(
                     type='SpatialBorderLoss', loss_weight=0.05),
                 loss_spatial_refine=dict(
                     type='SpatialBorderLoss', loss_weight=0.1),
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 center_init=True,
                 version='oc',
                 top_ratio=0.4,
                 init_qua_weight=0.2,
                 ori_qua_weight=0.3,
                 poc_qua_weight=0.1,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='reppoints_cls_out',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(OrientedRepPointsHead, self).__init__(init_cfg)
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.center_init = center_init

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(
            self.point_strides, offset=0.)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(
                self.train_cfg.refine.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_spatial_init = build_loss(loss_spatial_init)
        self.loss_spatial_refine = build_loss(loss_spatial_refine)
        self.init_qua_weight = init_qua_weight
        self.ori_qua_weight = ori_qua_weight
        self.poc_qua_weight = poc_qua_weight
        self.top_ratio = top_ratio
        self.version = version
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1,
                                               self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def forward(self, feats):
        """Forward function."""
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature map of a single FPN level.
        Args:
            x (torch.tensor): single-level feature map sizes.

        Returns:
            cls_out (torch.tensor): classification score prediction
            pts_out_init (torch.tensor): initial point sets prediction
            pts_out_refine (torch.tensor): refined point sets prediction
            base_feat: single-level feature as the basic feature map
        """
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        points_init = 0
        cls_feat = x
        pts_feat = x
        base_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        pts_out_refine = pts_out_refine + pts_out_init.detach()

        return cls_out, pts_out_init, pts_out_refine, base_feat

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl, _ in enumerate(self.point_strides):
            pts_lvl = []
            for i_img, _ in enumerate(center_list):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def sampling_points(self, polygons, points_num, device):
        """Sample edge points for polygon.

        Args:
            polygons (torch.tensor): polygons with shape (N, 8)
            points_num (int): number of sampling points for each polygon edge.
                              10 by default.

        Returns:
            sampling_points (torch.tensor): sampling points with shape (N,
                             points_num*4, 2)
        """
        polygons_xs, polygons_ys = polygons[:, 0::2], polygons[:, 1::2]
        ratio = torch.linspace(0, 1, points_num).to(device).repeat(
            polygons.shape[0], 1)

        edge_pts_x = []
        edge_pts_y = []
        for i in range(4):
            if i < 3:
                points_x = ratio * polygons_xs[:, i + 1:i + 2] + (
                    1 - ratio) * polygons_xs[:, i:i + 1]
                points_y = ratio * polygons_ys[:, i + 1:i + 2] + (
                    1 - ratio) * polygons_ys[:, i:i + 1]
            else:
                points_x = ratio * polygons_xs[:, 0].unsqueeze(1) + (
                    1 - ratio) * polygons_xs[:, i].unsqueeze(1)
                points_y = ratio * polygons_ys[:, 0].unsqueeze(1) + (
                    1 - ratio) * polygons_ys[:, i].unsqueeze(1)

            edge_pts_x.append(points_x)
            edge_pts_y.append(points_y)

        sampling_points_x = torch.cat(edge_pts_x, dim=1).unsqueeze(dim=2)
        sampling_points_y = torch.cat(edge_pts_y, dim=1).unsqueeze(dim=2)
        sampling_points = torch.cat([sampling_points_x, sampling_points_y],
                                    dim=2)

        return sampling_points

    def get_adaptive_points_feature(self, features, pt_locations, stride):
        """Get the points features from the locations of predicted points.

        Args:
            features (torch.tensor): base feature with shape (B,C,W,H)
            pt_locations (torch.tensor): locations of points in each point set
                     with shape (B, N_points_set(number of point set),
                     N_points(number of points in each point set) *2)
        Returns:
            tensor: sampling features with (B, C, N_points_set, N_points)
        """

        h = features.shape[2] * stride
        w = features.shape[3] * stride

        pt_locations = pt_locations.view(pt_locations.shape[0],
                                         pt_locations.shape[1], -1, 2).clone()
        pt_locations[..., 0] = pt_locations[..., 0] / (w / 2.) - 1
        pt_locations[..., 1] = pt_locations[..., 1] / (h / 2.) - 1

        batch_size = features.size(0)
        sampled_features = torch.zeros([
            pt_locations.shape[0],
            features.size(1),
            pt_locations.size(1),
            pt_locations.size(2)
        ]).to(pt_locations.device)

        for i in range(batch_size):
            feature = nn.functional.grid_sample(features[i:i + 1],
                                                pt_locations[i:i + 1])[0]
            sampled_features[i] = feature

        return sampled_features,

    def feature_cosine_similarity(self, points_features):
        """Compute the points features similarity for points-wise correlation.

        Args:
            points_features (torch.tensor): sampling point feature with
                     shape (N_pointsets, N_points, C)
        Returns:
            max_correlation: max feature similarity in each point set with
                     shape (N_points_set, N_points, C)
        """

        mean_points_feats = torch.mean(points_features, dim=1, keepdim=True)
        norm_pts_feats = torch.norm(
            points_features, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)
        norm_mean_pts_feats = torch.norm(
            mean_points_feats, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)

        unity_points_features = points_features / norm_pts_feats
        unity_mean_points_feats = mean_points_feats / norm_mean_pts_feats

        cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
        feats_similarity = 1.0 - cos_similarity(unity_points_features,
                                                unity_mean_points_feats)

        max_correlation, _ = torch.max(feats_similarity, dim=1)

        return max_correlation

    def pointsets_quality_assessment(self, pts_features, cls_score,
                                     pts_pred_init, pts_pred_refine, label,
                                     bbox_gt, label_weight, bbox_weight,
                                     pos_inds):
        """Assess the quality of each point set from the classification,
        localization, orientation, and point-wise correlation based on
        the assigned point sets samples.
        Args:
            pts_features (torch.tensor): points features with shape (N, 9, C)
            cls_score (torch.tensor): classification scores with
                        shape (N, class_num)
            pts_pred_init (torch.tensor): initial point sets prediction with
                        shape (N, 9*2)
            pts_pred_refine (torch.tensor): refined point sets prediction with
                        shape (N, 9*2)
            label (torch.tensor): gt label with shape (N)
            bbox_gt(torch.tensor): gt bbox of polygon with shape (N, 8)
            label_weight (torch.tensor): label weight with shape (N)
            bbox_weight (torch.tensor): box weight with shape (N)
            pos_inds (torch.tensor): the  inds of  positive point set samples

        Returns:
            qua (torch.tensor) : weighted quality values for positive
                                 point set samples.
        """
        device = cls_score.device
        pos_scores = cls_score[pos_inds]
        pos_pts_pred_init = pts_pred_init[pos_inds]
        pos_pts_pred_refine = pts_pred_refine[pos_inds]
        pos_pts_refine_features = pts_features[pos_inds]
        pos_bbox_gt = bbox_gt[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_bbox_weight = bbox_weight[pos_inds]

        # quality of point-wise correlation
        qua_poc = self.poc_qua_weight * self.feature_cosine_similarity(
            pos_pts_refine_features)

        qua_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        polygons_pred_init = min_area_polygons(pos_pts_pred_init)
        polygons_pred_refine = min_area_polygons(pos_pts_pred_refine)
        sampling_pts_pred_init = self.sampling_points(
            polygons_pred_init, 10, device=device)
        sampling_pts_pred_refine = self.sampling_points(
            polygons_pred_refine, 10, device=device)
        sampling_pts_gt = self.sampling_points(pos_bbox_gt, 10, device=device)

        # quality of orientation
        qua_ori_init = self.ori_qua_weight * ChamferDistance2D(
            sampling_pts_gt, sampling_pts_pred_init)
        qua_ori_refine = self.ori_qua_weight * ChamferDistance2D(
            sampling_pts_gt, sampling_pts_pred_refine)

        # quality of localization
        qua_loc_init = self.loss_bbox_refine(
            pos_pts_pred_init,
            pos_bbox_gt,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        qua_loc_refine = self.loss_bbox_refine(
            pos_pts_pred_refine,
            pos_bbox_gt,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        # quality of classification
        qua_cls = qua_cls.sum(-1)

        # weighted inti-stage and refine-stage
        qua = qua_cls + self.init_qua_weight * (
            qua_loc_init + qua_ori_init) + (1.0 - self.init_qua_weight) * (
                qua_loc_refine + qua_ori_refine) + qua_poc

        return qua,

    def dynamic_pointset_samples_selection(self,
                                           quality,
                                           label,
                                           label_weight,
                                           bbox_weight,
                                           pos_inds,
                                           pos_gt_inds,
                                           num_proposals_each_level=None,
                                           num_level=None):
        """The dynamic top k selection of point set samples based on the
        quality assessment values.

        Args:
            quality (torch.tensor): the quality values of positive
                                    point set samples
            label (torch.tensor): gt label with shape (N)
            bbox_gt(torch.tensor): gt bbox of polygon with shape (N, 8)
            label_weight (torch.tensor): label weight with shape (N)
            bbox_weight (torch.tensor): box weight with shape (N)
            pos_inds (torch.tensor): the inds of  positive point set samples
            num_proposals_each_level (list[int]): proposals number of
                                    each level
            num_level (int): the level number
        Returns:
            label: gt label with shape (N)
            label_weight: label weight with shape (N)
            bbox_weight: box weight with shape (N)
            num_pos (int): the number of selected positive point samples
                           with high-qualty
            pos_normalize_term (torch.tensor): the corresponding positive
                             normalize term
        """

        if len(pos_inds) == 0:
            return label, label_weight, bbox_weight, 0, torch.tensor(
                []).type_as(bbox_weight)

        num_gt = pos_gt_inds.max()
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)

        pos_inds_after_select = []
        ignore_inds_after_select = []

        for gt_ind in range(num_gt):
            pos_inds_select = []
            pos_loss_select = []
            gt_mask = pos_gt_inds == (gt_ind + 1)
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                value, topk_inds = quality[level_gt_mask].topk(
                    min(level_gt_mask.sum(), 6), largest=False)
                pos_inds_select.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_select.append(value)
            pos_inds_select = torch.cat(pos_inds_select)
            pos_loss_select = torch.cat(pos_loss_select)

            if len(pos_inds_select) < 2:
                pos_inds_after_select.append(pos_inds_select)
                ignore_inds_after_select.append(pos_inds_select.new_tensor([]))

            else:
                pos_loss_select, sort_inds = pos_loss_select.sort(
                )  # small to large
                pos_inds_select = pos_inds_select[sort_inds]
                # dynamic top k
                topk = math.ceil(pos_loss_select.shape[0] * self.top_ratio)
                pos_inds_select_topk = pos_inds_select[:topk]
                pos_inds_after_select.append(pos_inds_select_topk)
                ignore_inds_after_select.append(
                    pos_inds_select_topk.new_tensor([]))

        pos_inds_after_select = torch.cat(pos_inds_after_select)
        ignore_inds_after_select = torch.cat(ignore_inds_after_select)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_select).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_select] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_select)

        pos_level_mask_after_select = []
        for i in range(num_level):
            mask = (pos_inds_after_select >= inds_level_interval[i]) & (
                pos_inds_after_select < inds_level_interval[i + 1])
            pos_level_mask_after_select.append(mask)
        pos_level_mask_after_select = torch.stack(pos_level_mask_after_select,
                                                  0).type_as(label)
        pos_normalize_term = pos_level_mask_after_select * (
            self.point_base_scale *
            torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[
            pos_normalize_term > 0].type_as(bbox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_select)

        return label, label_weight, bbox_weight, num_pos, pos_normalize_term

    def init_loss_single(self, pts_pred_init, bbox_gt_init, bbox_weights_init,
                         stride):
        """Single initial stage loss function."""
        normalize_term = self.point_base_scale * stride

        bbox_gt_init = bbox_gt_init.reshape(-1, 8)
        bbox_weights_init = bbox_weights_init.reshape(-1)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        pos_ind_init = (bbox_weights_init > 0).nonzero(
            as_tuple=False).reshape(-1)

        pts_pred_init_norm = pts_pred_init[pos_ind_init]
        bbox_gt_init_norm = bbox_gt_init[pos_ind_init]
        bbox_weights_pos_init = bbox_weights_init[pos_ind_init]

        loss_pts_init = self.loss_bbox_init(
            pts_pred_init_norm / normalize_term,
            bbox_gt_init_norm / normalize_term, bbox_weights_pos_init)

        loss_border_init = self.loss_spatial_init(
            pts_pred_init_norm.reshape(-1, 2 * self.num_points) /
            normalize_term,
            bbox_gt_init_norm / normalize_term,
            bbox_weights_pos_init,
            avg_factor=None)

        return loss_pts_init, loss_border_init

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             overlaps,
                             stage='init',
                             unmap_outputs=True):
        """Single point target function for initial and refine stage."""
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight

        # convert gt from obb to poly
        gt_bboxes = obb2poly(gt_bboxes, self.version)

        assign_result = assigner.assign(proposals, gt_bboxes, overlaps,
                                        gt_bboxes_ignore,
                                        None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)

        gt_inds = assign_result.gt_inds
        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 8])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros(num_valid_proposals)

        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)
            gt_inds = unmap(gt_inds, num_total_proposals, inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, gt_inds,
                sampling_result)

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals in initial stage.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of \
                    each level.
                - proposal_weights_list (list[Tensor]): Proposal weights of \
                    each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * len(proposals_list)
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list, all_gt_inds,
         sampling_result) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             all_overlaps_rotate_list,
             stage=stage,
             unmap_outputs=unmap_outputs)

        if stage == 'init':
            # no valid points
            if any([labels is None for labels in all_labels]):
                return None
            # sampled points of all images
            num_total_pos = sum(
                [max(inds.numel(), 1) for inds in pos_inds_list])
            num_total_neg = sum(
                [max(inds.numel(), 1) for inds in neg_inds_list])
            labels_list = images_to_levels(all_labels, num_level_proposals)
            label_weights_list = images_to_levels(all_label_weights,
                                                  num_level_proposals)
            bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
            proposals_list = images_to_levels(all_proposals,
                                              num_level_proposals)
            proposal_weights_list = images_to_levels(all_proposal_weights,
                                                     num_level_proposals)

            return (labels_list, label_weights_list, bbox_gt_list,
                    proposals_list, proposal_weights_list, num_total_pos,
                    num_total_neg, None)

        else:
            pos_inds = []
            pos_gt_index = []
            for i, single_labels in enumerate(all_labels):
                pos_mask = (0 <= single_labels) & (
                    single_labels < self.num_classes)
                pos_inds.append(pos_mask.nonzero(as_tuple=False).view(-1))
                pos_gt_index.append(
                    all_gt_inds[i][pos_mask.nonzero(as_tuple=False).view(-1)])

            return (all_labels, all_label_weights, all_bbox_gt, all_proposals,
                    all_proposal_weights, pos_inds, pos_gt_index)

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             base_features,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Loss function of OrientedRepPoints head."""

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        device = cls_scores[0].device

        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)

        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)

        num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                    for featmap in cls_scores]

        num_level = len(featmap_sizes)
        assert num_level == len(pts_coordinate_preds_init)

        candidate_list = center_list

        cls_reg_targets_init = self.get_targets(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init, _) = cls_reg_targets_init

        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)

        refine_points_features, = multi_apply(self.get_adaptive_points_feature,
                                              base_features,
                                              pts_coordinate_preds_refine,
                                              self.point_strides)
        features_pts_refine = levels_to_images(refine_points_features)
        features_pts_refine = [
            item.reshape(-1, self.num_points, item.shape[-1])
            for item in features_pts_refine
        ]

        points_list = []
        for i_img, center in enumerate(center_list):
            points = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(
                    points_preds_init_.shape[0], -1,
                    *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(
                    0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                points.append(
                    points_center +
                    points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)

        cls_reg_targets_refine = self.get_targets(
            points_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='refine',
            label_channels=label_channels)

        (labels_list, label_weights_list, bbox_gt_list_refine, _,
         bbox_weights_list_refine, pos_inds_list_refine,
         pos_gt_index_list_refine) = cls_reg_targets_refine

        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]

        pts_coordinate_preds_init_img = levels_to_images(
            pts_coordinate_preds_init, flatten=True)
        pts_coordinate_preds_init_img = [
            item.reshape(-1, 2 * self.num_points)
            for item in pts_coordinate_preds_init_img
        ]

        pts_coordinate_preds_refine_img = levels_to_images(
            pts_coordinate_preds_refine, flatten=True)
        pts_coordinate_preds_refine_img = [
            item.reshape(-1, 2 * self.num_points)
            for item in pts_coordinate_preds_refine_img
        ]

        with torch.no_grad():

            quality_assess_list, = multi_apply(
                self.pointsets_quality_assessment, features_pts_refine,
                cls_scores, pts_coordinate_preds_init_img,
                pts_coordinate_preds_refine_img, labels_list,
                bbox_gt_list_refine, label_weights_list,
                bbox_weights_list_refine, pos_inds_list_refine)

            labels_list, label_weights_list, bbox_weights_list_refine, \
                num_pos, pos_normalize_term = multi_apply(
                    self.dynamic_pointset_samples_selection,
                    quality_assess_list,
                    labels_list,
                    label_weights_list,
                    bbox_weights_list_refine,
                    pos_inds_list_refine,
                    pos_gt_index_list_refine,
                    num_proposals_each_level=num_proposals_each_level,
                    num_level=num_level
                )
            num_pos = sum(num_pos)

        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        pts_preds_refine = torch.cat(pts_coordinate_preds_refine_img, 0).view(
            -1, pts_coordinate_preds_refine_img[0].size(-1))

        labels = torch.cat(labels_list, 0).view(-1)
        labels_weight = torch.cat(label_weights_list, 0).view(-1)
        bbox_gt_refine = torch.cat(bbox_gt_list_refine,
                                   0).view(-1, bbox_gt_list_refine[0].size(-1))
        bbox_weights_refine = torch.cat(bbox_weights_list_refine, 0).view(-1)
        pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)
        pos_inds_flatten = ((0 <= labels) &
                            (labels < self.num_classes)).nonzero(
                                as_tuple=False).reshape(-1)

        assert len(pos_normalize_term) == len(pos_inds_flatten)

        if num_pos:
            losses_cls = self.loss_cls(
                cls_scores, labels, labels_weight, avg_factor=num_pos)
            pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
            pos_bbox_gt_refine = bbox_gt_refine[pos_inds_flatten]

            pos_bbox_weights_refine = bbox_weights_refine[pos_inds_flatten]
            losses_pts_refine = self.loss_bbox_refine(
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                pos_bbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_bbox_weights_refine)

            loss_border_refine = self.loss_spatial_refine(
                pos_pts_pred_refine.reshape(-1, 2 * self.num_points) /
                pos_normalize_term.reshape(-1, 1),
                pos_bbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_bbox_weights_refine,
                avg_factor=None)

        else:
            losses_cls = cls_scores.sum() * 0
            losses_pts_refine = pts_preds_refine.sum() * 0
            loss_border_refine = pts_preds_refine.sum() * 0

        losses_pts_init, loss_border_init = multi_apply(
            self.init_loss_single, pts_coordinate_preds_init,
            bbox_gt_list_init, bbox_weights_list_init, self.point_strides)

        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine,
            'loss_spatial_init': loss_border_init,
            'loss_spatial_refine': loss_border_refine
        }
        return loss_dict_all

    @force_fp32(apply_to=('cls_scores', 'pts_preds_init', 'pts_preds_refine'))
    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   base_feats,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            pts_preds_init (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 18D-tensor, has shape
                (batch_size, num_points * 2, H, W).
            pts_preds_refine (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 18D-tensor, has shape
                (batch_size, num_points * 2, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(pts_preds_refine)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].device,
            device=cls_scores[0].device)

        result_list = []

        for img_id, _ in enumerate(img_metas):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            point_pred_list = select_single_mlvl(pts_preds_refine, img_id)

            results = self._get_bboxes_single(cls_score_list, point_pred_list,
                                              mlvl_priors, img_meta, cfg,
                                              rescale, with_nms, **kwargs)
            result_list.append(results)

        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           point_pred_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RepPoints head does not need
                this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (cx, cy, w, h, a) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(point_pred_list)
        scale_factor = img_meta['scale_factor']

        mlvl_bboxes = []
        mlvl_scores = []
        for level_idx, (cls_score, points_pred, points) in enumerate(
                zip(cls_score_list, point_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            points_pred = points_pred.permute(1, 2, 0).reshape(
                -1, 2 * self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            pts_pred = points_pred.reshape(-1, self.num_points, 2)
            pts_pred_offsety = pts_pred[:, :, 0::2]
            pts_pred_offsetx = pts_pred[:, :, 1::2]
            pts_pred = torch.cat([pts_pred_offsetx, pts_pred_offsety],
                                 dim=2).reshape(-1, 2 * self.num_points)

            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts_pred * self.point_strides[level_idx] + pts_pos_center

            polys = min_area_polygons(pts)
            bboxes = poly2obb(polys, self.version)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)

        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes[..., :4].new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            raise NotImplementedError
