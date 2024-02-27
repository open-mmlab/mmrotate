# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, is_norm
from mmdet.models.dense_heads import YOLOFHead
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, levels_to_images,
                                multi_apply, unmap)
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                                   cat_boxes)
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes

INF = 1e8


@MODELS.register_module()
class Point2RBoxYOLOFHead(YOLOFHead):
    """Detection Head of `Point2RBox <https://arxiv.org/abs/2311.14758>`_

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (list[int]): The number of input channels per scale.
        cls_num_convs (int): The number of convolutions of cls branch.
           Defaults to 2.
        reg_num_convs (int): The number of convolutions of reg branch.
           Defaults to 4.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='BN', requires_grad=True)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: List[int],
                 num_cls_convs: int = 2,
                 num_reg_convs: int = 4,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 use_bbox_hdr: bool = False,
                 use_transform_ss: bool = True,
                 use_objectness: bool = True,
                 full_supervised: bool = False,
                 agnostic_cls: list = [1, 9, 11],
                 square_cls: list = [0],
                 synthetic_pos_weight: float = 0.1,
                 loss_point: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=0.1),
                 angle_coder: ConfigType = dict(
                     type='PSCCoder',
                     angle_version='le90',
                     dual_freq=False,
                     thr_mod=0),
                 loss_angle: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=0.1),
                 loss_ratio: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=1.0),
                 loss_symmetry_ss: ConfigType = dict(
                     type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
                 loss_scale_ss: ConfigType = dict(
                     type='mmdet.GIoULoss', loss_weight=0.05),
                 **kwargs) -> None:
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_cfg = norm_cfg
        self.use_bbox_hdr = use_bbox_hdr
        self.use_transform_ss = use_transform_ss
        self.use_objectness = use_objectness
        self.full_supervised = full_supervised
        self.agnostic_cls = agnostic_cls
        self.square_cls = square_cls
        self.synthetic_pos_weight = synthetic_pos_weight
        self.angle_coder = TASK_UTILS.build(angle_coder)
        super().__init__(
            num_classes=num_classes, in_channels=in_channels, **kwargs)
        self.loss_point = MODELS.build(loss_point)
        self.loss_angle = MODELS.build(loss_angle)
        self.loss_ratio = MODELS.build(loss_ratio)
        self.loss_symmetry_ss = MODELS.build(loss_symmetry_ss)
        self.loss_scale_ss = MODELS.build(loss_scale_ss)

    def _init_layers(self) -> None:
        cls_subnet = []
        bbox_subnet = []
        ang_subnet = []
        for i in range(self.num_cls_convs):
            cls_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        for i in range(self.num_reg_convs):
            bbox_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        for i in range(self.num_reg_convs):
            ang_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.ang_subnet = nn.Sequential(*ang_subnet)
        self.cls_score = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.cls_score_f = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_cent_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * 2,  # CenterXY
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_size_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * 2 *
            (5 if self.use_bbox_hdr else 1),  # SizeXY
            kernel_size=3,
            stride=1,
            padding=1)
        if self.use_objectness:
            self.object_pred = nn.Conv2d(
                self.in_channels,
                self.num_base_priors,
                kernel_size=3,
                stride=1,
                padding=1)
        self.angle_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.angle_coder.encode_size,
            kernel_size=3,
            stride=1,
            padding=1)
        self.ratio_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors,
            kernel_size=3,
            stride=1,
            padding=1)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        # Use prior in model initialization to improve stability
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.cls_score.bias, bias_cls)
        torch.nn.init.constant_(self.cls_score_f.bias, bias_cls)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                normalized_cls_score (Tensor): Normalized Cls scores for a \
                    single scale level, the channels number is \
                    num_base_priors * num_classes.
                bbox_reg (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        cls_feat = self.cls_subnet(x)
        cls_score = self.cls_score(cls_feat)
        cls_score_f = self.cls_score_f(cls_feat)
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(x)
        ang_feat = self.ang_subnet(x)

        if self.use_bbox_hdr:
            r = self.bbox_size_pred(reg_feat).sigmoid() * 4 - 2
            r = r.view(r.shape[0], 5, -1, *r.shape[2:])
            w = torch.softmax(-r.abs(), 1)
            o = r.new_tensor((-2, -1, 0, 1, 2))[None, :, None, None, None]
            bbox_size_reg = (w * (r + o)).sum(1) * 2
        else:
            bbox_size_reg = self.bbox_size_pred(reg_feat)

        bbox_cent_reg = self.bbox_cent_pred(reg_feat)
        angle_reg = self.angle_pred(ang_feat)
        ratio_reg = self.ratio_pred(ang_feat).sigmoid()

        bbox_reg = (bbox_cent_reg, bbox_size_reg, ratio_reg, angle_reg)
        bbox_reg = [
            x.view(N, self.num_base_priors, -1, H, W) for x in bbox_reg
        ]
        bbox_reg = torch.cat(bbox_reg, 2).view(N, -1, H, W)

        # implicit objectness
        if self.use_objectness:
            objectness = self.object_pred(reg_feat)
            objectness = objectness.view(N, -1, 1, H, W)
            normalized_cls_score = cls_score + objectness - torch.log(
                1. + torch.clamp(cls_score.exp(), max=INF) +
                torch.clamp(objectness.exp(), max=INF))
            normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        else:
            normalized_cls_score = cls_score.view(N, -1, H, W)

        return normalized_cls_score, cls_score_f, bbox_reg

    def obb2xyxy(self, obb):
        w = obb[:, 2::5]
        h = obb[:, 3::5]
        a = obb[:, 4::5]
        cosa = torch.cos(a).abs()
        sina = torch.sin(a).abs()
        hbbox_w = cosa * w + sina * h
        hbbox_h = sina * w + cosa * h
        dx = obb[..., 0]
        dy = obb[..., 1]
        dw = hbbox_w.reshape(-1)
        dh = hbbox_h.reshape(-1)
        x1 = dx - dw / 2
        y1 = dy - dh / 2
        x2 = dx + dw / 2
        y2 = dy + dh / 2
        return torch.stack((x1, y1, x2, y2), -1)

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            cls_scores_f: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        assert len(cls_scores) == 1
        assert self.prior_generator.num_levels == 1

        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        # The output level is always 1
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]

        # cls_scores_list = levels_to_images(cls_scores)
        cls_scores_f_list = levels_to_images(cls_scores_f)
        bbox_preds_list = levels_to_images(bbox_preds)

        cls_reg_targets = self.get_targets(
            cls_scores_f_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        if cls_reg_targets is None:
            return None
        (batch_labels, batch_label_weights, avg_factor, bbox_weights,
         point_weights, pos_pred_boxes, target_boxes, target_labels,
         target_bids) = cls_reg_targets

        flatten_labels = batch_labels.reshape(-1)
        batch_label_weights = batch_label_weights.reshape(-1)
        cls_score = cls_scores[0].permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)
        cls_score_f = cls_scores_f[0].permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)

        point_f_mask = batch_label_weights == self.synthetic_pos_weight
        label_weight = batch_label_weights.clone()
        label_weight[point_f_mask] = 0
        avg_factor_point = max(avg_factor - point_f_mask.sum().item(), 1)
        avg_factor_point = reduce_mean(
            torch.tensor(avg_factor_point, dtype=torch.float,
                         device=device)).item()

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        # classification loss
        if self.full_supervised:
            loss_cls = self.loss_cls(
                cls_score,
                flatten_labels,
                batch_label_weights,
                avg_factor=avg_factor)
            loss_cls_f = 0 * loss_cls
        else:
            loss_cls = self.loss_cls(
                cls_score,
                flatten_labels,
                label_weight,
                avg_factor=avg_factor_point)
            loss_cls_f = self.loss_cls(
                cls_score_f,
                flatten_labels,
                batch_label_weights,
                avg_factor=avg_factor)

        # regression loss
        if pos_pred_boxes.shape[0] == 0:
            # no pos sample
            loss_bbox = pos_pred_boxes.sum() * 0
            loss_angle = pos_pred_boxes.sum() * 0
            loss_ratio = pos_pred_boxes.sum() * 0
            loss_point = pos_pred_boxes.sum() * 0
            loss_scale_ss = pos_pred_boxes.sum() * 0
            loss_symmetry_ss = pos_pred_boxes.sum() * 0
        else:
            target_boxes = RotatedBoxes(target_boxes).regularize_boxes('le90')
            if self.agnostic_cls:
                agnostic_mask = torch.stack([
                    target_labels == c for c in self.agnostic_cls
                ]).sum(0).bool()
            else:
                agnostic_mask = target_labels < 0
            target_boxes[agnostic_mask, 4] = 0
            if self.square_cls:
                square_mask = torch.stack([
                    target_labels == c for c in self.square_cls
                ]).sum(0).bool()
            else:
                square_mask = target_labels < 0
            target_boxes[square_mask, 4] = 0

            pos_pred_xyxy = pos_pred_boxes[:, :4]
            target_xyxy = self.obb2xyxy(target_boxes)
            loss_bbox = self.loss_bbox(
                pos_pred_xyxy,
                target_xyxy,
                bbox_weights.float(),
                avg_factor=bbox_weights.sum())

            angle_weights = bbox_weights.clone().float()
            angle_weights[agnostic_mask] = 0
            angle_weights[square_mask] = 0
            pos_pred_angle = pos_pred_boxes[:, 5:]
            target_angle = self.angle_coder.encode(target_boxes[:, 4:])
            loss_angle = self.loss_angle(
                pos_pred_angle,
                target_angle,
                angle_weights[:, None],
                avg_factor=angle_weights.sum())

            pos_pred_ratio = pos_pred_boxes[:, 4]
            target_ratio = target_boxes[:, 3] / (target_boxes[:, 2] + 1e-5)
            loss_ratio = self.loss_ratio(
                pos_pred_ratio,
                target_ratio,
                angle_weights.float(),
                avg_factor=angle_weights.sum())

            pos_pred_cxcywh = bbox_xyxy_to_cxcywh(pos_pred_boxes[:, :4])
            pos_pred_cen = pos_pred_cxcywh[:, 0:2]
            target_cen = target_boxes[:, 0:2]
            point_valid = (pos_pred_cen - target_cen).abs().sum(1) < 32
            point_weights *= point_valid
            loss_point = self.loss_point(
                pos_pred_cen / 16,
                target_cen / 16,
                point_weights.float()[:, None],
                avg_factor=point_weights.sum())

            if self.use_transform_ss and loss_bbox.item(
            ) < 0.5 and loss_angle.item() < 0.2:
                # Self-supervision
                # Calculate SS only for point annotations
                target_bids[~point_weights] = -1
                # print(f'{target_bids[point_weights] = }')

                # Aggregate the same bbox based on their identical bid
                bid, idx = torch.unique(target_bids, return_inverse=True)
                pair_bid_targets = torch.empty_like(bid).index_reduce_(
                    0, idx, target_bids, 'mean', include_self=False)

                # Generate a mask to eliminate bboxes without correspondence
                # (bcnt is supposed to be 3, for ori, rot, and flp)
                _, bidx, bcnt = torch.unique(
                    pair_bid_targets.long(),
                    return_inverse=True,
                    return_counts=True)
                bmsk = bcnt[bidx] == 2

                # print(pair_bid_targets)
                b_sca = (pair_bid_targets % 1 > 0.7).sum() > 0

                # The reduce all sample points of each object
                pair_box_target = torch.empty_like(bid).index_reduce_(
                    0, idx, target_boxes[:, 2], 'mean',
                    include_self=False)[bmsk].view(-1, 2)
                pair_box_preds = torch.empty(
                    *bid.shape, pos_pred_cxcywh.shape[-1],
                    device=bid.device).index_reduce_(
                        0, idx, pos_pred_cxcywh, 'mean',
                        include_self=False)[bmsk].view(
                            -1, 2, pos_pred_cxcywh.shape[-1])

                ori_box = pair_box_preds[:, 0]
                trs_box = pair_box_preds[:, 1]

                if b_sca:
                    sca = (pair_box_target[:, 1] /
                           pair_box_target[:, 0]).mean()
                    ori_box *= sca

                    # Must limit the center and size range in ss
                    ss_weight_cen = (ori_box[:, :2] -
                                     trs_box[:, :2]).abs().sum(1) < 32
                    ss_weight_wh0 = (ori_box[:, 2:] +
                                     trs_box[:, 2:]).sum(1) > 12 * 4
                    ss_weight_wh1 = (ori_box[:, 2:] +
                                     trs_box[:, 2:]).sum(1) < 512 * 4
                    ss_weight = ss_weight_cen * ss_weight_wh0 * ss_weight_wh1
                    if len(ori_box):
                        loss_scale_ss = self.loss_scale_ss(
                            bbox_cxcywh_to_xyxy(ori_box),
                            bbox_cxcywh_to_xyxy(trs_box), ss_weight)
                    else:
                        loss_scale_ss = pos_pred_cxcywh.sum() * 0
                    loss_symmetry_ss = pos_pred_angle.sum() * 0
                else:
                    b_flp = (pair_bid_targets % 1 > 0.5).sum() > 0

                    # The reduce all sample points of each object
                    pair_angle_targets = torch.empty_like(bid).index_reduce_(
                        0, idx, target_boxes[:, 4], 'mean',
                        include_self=False)[bmsk].view(-1, 2)
                    pair_angle_preds = torch.empty(
                        *bid.shape,
                        pos_pred_angle.shape[-1],
                        device=bid.device).index_reduce_(
                            0, idx, pos_pred_angle, 'mean',
                            include_self=False)[bmsk].view(
                                -1, 2, pos_pred_angle.shape[-1])

                    pair_angle_preds = self.angle_coder.decode(
                        pair_angle_preds, keepdim=False)

                    # Eliminate invalid pairs
                    img_shape = batch_img_metas[0]['img_shape']
                    if b_flp:
                        flp_box = ori_box[:, :2]
                        flp_box[:, 1] = img_shape[0] - flp_box[:, 1]
                        ss_weight = (flp_box[:, :2] -
                                     trs_box[:, :2]).abs().sum(1) < 32
                        d_ang = pair_angle_preds[:, 0] + pair_angle_preds[:, 1]
                    else:
                        a = pair_angle_targets[:, 0] - pair_angle_targets[:, 1]
                        cosa = torch.cos(a)
                        sina = torch.sin(a)
                        m = torch.stack((cosa, sina, -sina, cosa),
                                        -1).view(-1, 2, 2)
                        rot_box = torch.bmm(
                            m, (ori_box[:, :2] - img_shape[0] /
                                2)[..., None])[:, :, 0] + img_shape[0] / 2
                        ss_weight = (rot_box[:, :2] -
                                     trs_box[:, :2]).abs().sum(1) < 32
                        d_ang = (pair_angle_preds[:, 0] -
                                 pair_angle_preds[:, 1]) - (
                                     pair_angle_targets[:, 0] -
                                     pair_angle_targets[:, 1])

                    # Eliminate agnostic objects
                    if self.agnostic_cls:
                        pair_labels = torch.empty(
                            bid.shape,
                            dtype=target_labels.dtype,
                            device=bid.device).index_reduce_(
                                0,
                                idx,
                                target_labels,
                                'mean',
                                include_self=False)[bmsk].view(-1, 2)[:, 0]
                        pair_agnostic_mask = torch.stack([
                            pair_labels == c for c in self.agnostic_cls
                        ]).sum(0).bool()
                        ss_weight[pair_agnostic_mask] = 0

                    d_ang = (d_ang + torch.pi / 2) % torch.pi - torch.pi / 2

                    loss_scale_ss = pos_pred_cxcywh.sum() * 0
                    if len(d_ang):
                        loss_symmetry_ss = self.loss_symmetry_ss(
                            d_ang, torch.zeros_like(d_ang), ss_weight)
                    else:
                        loss_symmetry_ss = pos_pred_angle.sum() * 0
            else:
                loss_scale_ss = pos_pred_cxcywh.sum() * 0
                loss_symmetry_ss = pos_pred_angle.sum() * 0

        return dict(
            loss_cls=loss_cls,
            loss_cls_f=loss_cls_f,
            loss_bbox=loss_bbox,
            loss_angle=loss_angle,
            loss_ratio=loss_ratio,
            loss_point=loss_point,
            loss_scale_ss=loss_scale_ss,
            loss_symmetry_ss=loss_symmetry_ss)

    def get_targets(self,
                    cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    anchor_list: List[Tensor],
                    valid_flag_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores_list (list[Tensor])： Classification scores of
                each image. each is a 4D-tensor, the shape is
                (h * w, num_anchors * num_classes).
            bbox_preds_list (list[Tensor])： Bbox preds of each image.
                each is a 4D-tensor, the shape is (h * w, num_anchors * 4).
            anchor_list (list[Tensor]): Anchors of each image. Each element of
                is a tensor of shape (h * w * num_anchors, 4).
            valid_flag_list (list[Tensor]): Valid flags of each image. Each
               element of is a tensor of shape (h * w * num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - batch_labels (Tensor): Label of all images. Each element \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - batch_label_weights (Tensor): Label weights of all images \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        results = multi_apply(
            self._get_targets_single,
            cls_scores_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, pos_inds, neg_inds,
         sampling_results_list) = results[:5]
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        rest_results = list(results[5:])  # user-added return values

        batch_labels = torch.stack(all_labels, 0)
        batch_label_weights = torch.stack(all_label_weights, 0)

        res = (batch_labels, batch_label_weights, avg_factor)
        for i, rests in enumerate(rest_results):  # user-added return values
            rest_results[i] = torch.cat(rests, 0)

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            bbox_preds (Tensor): Bbox prediction of the image, which
                shape is (h * w ,4)
            flat_anchors (Tensor): Anchors of the image, which shape is
                (h * w * num_anchors ,4)
            valid_flags (Tensor): Valid flags of the image, which shape is
                (h * w * num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels (Tensor): Labels of image, which shape is
                    (h * w * num_anchors, ).
                label_weights (Tensor): Label weights of image, which shape is
                    (h * w * num_anchors, ).
                pos_inds (Tensor): Pos index of image.
                neg_inds (Tensor): Neg index of image.
                sampling_result (obj:`SamplingResult`): Sampling result.
                pos_bbox_weights (Tensor): The Weight of using to calculate
                    the bbox branch loss, which shape is (num, ).
                pos_predicted_boxes (Tensor): boxes predicted value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
                pos_target_boxes (Tensor): boxes target value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')

        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        dim = self.bbox_coder.encode_size + self.angle_coder.encode_size + 1
        bbox_preds = bbox_preds.reshape(-1, dim)
        bbox_preds = bbox_preds[inside_flags, :]

        ###
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_scores = cls_scores[inside_flags, :]

        # decoded bbox
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds[:, :4])
        decoder_bbox_preds = torch.cat((decoder_bbox_preds, bbox_preds[:, 4:]),
                                       -1)
        pred_instances = InstanceData(
            priors=anchors,
            decoder_priors=decoder_bbox_preds,
            cls_scores=cls_scores)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        pos_point_index = assign_result.get_extra_property('pos_point_index')
        pos_bbox_weights = assign_result.get_extra_property('pos_bbox_mask')
        pos_point_weights = assign_result.get_extra_property('pos_point_mask')
        pos_predicted_boxes = assign_result.get_extra_property(
            'pos_predicted_boxes')
        pos_target_boxes = assign_result.get_extra_property('target_boxes')
        pos_target_labels = assign_result.get_extra_property('target_labels')
        pos_target_bids = assign_result.get_extra_property('target_bids')

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = self.synthetic_pos_weight
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if pos_point_index is not None:
            label_weights[pos_point_index.reshape(-1)] = 1

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)

        return (labels, label_weights, pos_inds, neg_inds, sampling_result,
                pos_bbox_weights, pos_point_weights, pos_predicted_boxes,
                pos_target_boxes, pos_target_labels, pos_target_bids)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        cls_scores_f: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:

        result_list = super().predict_by_feat(cls_scores, bbox_preds,
                                              score_factors, batch_img_metas,
                                              cfg, rescale, with_nms)

        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

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
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = (
                self.bbox_coder.encode_size + self.angle_coder.encode_size + 1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(
            priors, bbox_pred[:, :4], max_shape=img_shape)
        bboxes = bbox_xyxy_to_cxcywh(bboxes)
        ratios = bbox_pred[:, 4:5].clamp(0.05, 1)
        angles = self.angle_coder.decode(bbox_pred[:, 5:], keepdim=True)

        labels = torch.cat(mlvl_labels)
        if self.agnostic_cls:
            agnostic_mask = torch.stack(
                [labels == c for c in self.agnostic_cls]).sum(0).bool()
        else:
            agnostic_mask = labels < 0
        if self.square_cls:
            square_mask = torch.stack([labels == c for c in self.square_cls
                                       ]).sum(0).bool()
        else:
            square_mask = labels < 0
        angles[agnostic_mask] = 0
        ratios[agnostic_mask] = 1
        ratios[square_mask] = 1

        cosa = torch.cos(angles).abs()
        sina = torch.sin(angles).abs()
        m = torch.stack(
            (ratios, -torch.ones_like(ratios), cosa, sina, sina, cosa),
            -1).view(-1, 3, 2)
        b = torch.cat((torch.zeros_like(bboxes[:, 2:3]), bboxes[:, 2:4]),
                      1)[..., None]
        wh = torch.linalg.lstsq(m, b).solution[:, :, 0]
        wh[square_mask] *= 1.4

        # For DIOR
        if self.cls_out_channels == 20:
            wh[labels == 0] *= 0.8
            wh[labels == 17] *= 1.2
            wh[labels == 13] *= 0.7
            wh[labels == 18] *= 0.7
            wh[labels == 19] *= 0.7

        bboxes = torch.cat((bboxes[:, 0:2], wh, angles), 1)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = labels
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        results = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        return results
