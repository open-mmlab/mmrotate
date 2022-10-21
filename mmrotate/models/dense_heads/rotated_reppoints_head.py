# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmcv.ops import min_area_polygons
from mmdet.models.dense_heads.reppoints_head import RepPointsHead
from mmdet.models.utils import filter_scores_and_topk, multi_apply, unmap
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes, qbox2rbox


@MODELS.register_module()
class RotatedRepPointsHead(RepPointsHead):
    """RotatedRepPoint head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        point_feat_channels (int): Number of channels of points features.
        num_points (int): Number of points.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Sequence[int]): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox_init (:obj:`ConfigDict` or dict): Config of initial points
            loss.
        loss_bbox_refine (:obj:`ConfigDict` or dict): Config of points loss in
            refinement.
        transform_method (str): The methods to transform RepPoints to qbbox,
            which cannot be 'moment' in here.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: W605

    def __init__(self, *args, **kwargs) -> None:
        # avoid register scope error.
        super().__init__(
            *args,
            loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
            bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
            **kwargs)

    def forward_single(self, x: Tensor) -> Tuple[Tensor]:
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        if self.training:
            return cls_out, pts_out_init, pts_out_refine
        else:
            return cls_out, pts_out_refine

    def _get_targets_single(self,
                            flat_proposals: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            gt_instances_ignore: InstanceData,
                            stage: str = 'init',
                            unmap_outputs: bool = True) -> tuple:
        """Compute corresponding GT box and classification targets for
        proposals.
        Args:
            flat_proposals (Tensor): Multi level points of a image.
            valid_flags (Tensor): Multi level valid flags of a image.
            gt_instances (InstanceData): It usually includes ``bboxes`` and
                ``labels`` attributes.
            gt_instances_ignore (InstanceData): It includes ``bboxes``
                attribute data that is ignored during training and testing.
            stage (str): 'init' or 'refine'. Generate target for
                init stage or refine stage. Defaults to 'init'.
            unmap_outputs (bool): Whether to map outputs back to
                the original set of anchors. Defaults to True.
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
        inside_flags = valid_flags
        if not inside_flags.any():
            raise ValueError(
                'There is no valid proposal inside the image boundary. Please '
                'check the image size.')
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]
        pred_instances = InstanceData(priors=proposals)

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight

        assign_result = assigner.assign(pred_instances, gt_instances,
                                        gt_instances_ignore)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

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
            bbox_gt[pos_inds, :] = sampling_result.pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(
                labels,
                num_total_proposals,
                inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, sampling_result)

    def loss_by_feat_single(self, cls_score: Tensor, pts_pred_init: Tensor,
                            pts_pred_refine: Tensor, labels: Tensor,
                            label_weights, bbox_gt_init: Tensor,
                            bbox_weights_init: Tensor, bbox_gt_refine: Tensor,
                            bbox_weights_refine: Tensor, stride: int,
                            avg_factor_init: int,
                            avg_factor_refine: int) -> Tuple[Tensor]:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.
        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_classes, h_i, w_i).
            pts_pred_init (Tensor): Points of shape
                (batch_size, h_i * w_i, num_points * 2).
            pts_pred_refine (Tensor): Points refined of shape
                (batch_size, h_i * w_i, num_points * 2).
            labels (Tensor): Ground truth class indices with shape
                (batch_size, h_i * w_i).
            label_weights (Tensor): Label weights of shape
                (batch_size, h_i * w_i).
            bbox_gt_init (Tensor): BBox regression targets in the init stage
                of shape (batch_size, h_i * w_i, 8).
            bbox_weights_init (Tensor): BBox regression loss weights in the
                init stage of shape (batch_size, h_i * w_i, 8).
            bbox_gt_refine (Tensor): BBox regression targets in the refine
                stage of shape (batch_size, h_i * w_i, 8).
            bbox_weights_refine (Tensor): BBox regression loss weights in the
                refine stage of shape (batch_size, h_i * w_i, 8).
            stride (int): Point stride.
            avg_factor_init (int): Average factor that is used to average
                the loss in the init stage.
            avg_factor_refine (int): Average factor that is used to average
                the loss in the refine stage.
        Returns:
            Tuple[Tensor]: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score.contiguous()
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor_refine)

        # init loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 8)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        bbox_weights_init = bbox_weights_init.reshape(-1)
        pos_ind_init = (bbox_weights_init > 0).nonzero(
            as_tuple=False).reshape(-1)
        pos_bbox_gt_init = bbox_gt_init[pos_ind_init]
        pos_pts_pred_init = pts_pred_init[pos_ind_init]
        pos_bbox_weights_init = bbox_weights_init[pos_ind_init]
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(
            pos_pts_pred_init / normalize_term,
            pos_bbox_gt_init / normalize_term,
            pos_bbox_weights_init,
            avg_factor=avg_factor_init)

        # refine loss
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 8)
        pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        bbox_weights_refine = bbox_weights_refine.reshape(-1)
        pos_ind_refine = (bbox_weights_refine > 0).nonzero(
            as_tuple=False).reshape(-1)
        pos_bbox_gt_refine = bbox_gt_refine[pos_ind_refine]
        pos_pts_pred_refine = pts_pred_refine[pos_ind_refine]
        pos_bbox_weights_refine = bbox_weights_refine[pos_ind_refine]
        loss_pts_refine = self.loss_bbox_refine(
            pos_pts_pred_refine / normalize_term,
            pos_bbox_gt_refine / normalize_term,
            pos_bbox_weights_refine,
            avg_factor=avg_factor_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        pts_preds_init: List[Tensor],
        pts_preds_refine: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, of shape (batch_size, num_classes, h, w).
            pts_preds_init (list[Tensor]): Points for each scale level, each is
                a 3D-tensor, of shape (batch_size, h_i * w_i, num_points * 2).
            pts_preds_refine (list[Tensor]): Points refined for each scale
                level, each is a 3D-tensor, of shape
                (batch_size, h_i * w_i, num_points * 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       batch_img_metas, device)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if self.train_cfg.init.assigner['type'] == 'ConvexAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            raise NotImplementedError
        cls_reg_targets_init = self.get_targets(
            proposals_list=candidate_list,
            valid_flag_list=valid_flag_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='init',
            return_sampling_results=False)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         avg_factor_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       batch_img_metas, device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(
                    points_preds_init_.shape[0], -1,
                    *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(
                    0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                bbox.append(
                    points_center +
                    points_shift[i_img].reshape(-1, 2 * self.num_points))
            bbox_list.append(bbox)
        cls_reg_targets_refine = self.get_targets(
            proposals_list=bbox_list,
            valid_flag_list=valid_flag_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='refine',
            return_sampling_results=False)
        (labels_list, label_weights_list, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine,
         avg_factor_refine) = cls_reg_targets_refine

        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            avg_factor_init=avg_factor_init,
            avg_factor_refine=avg_factor_refine)
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        return loss_dict_all

    # Same as base_dense_head/_get_bboxes_single except self._bbox_decode
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform outputs of a single image into bbox predictions.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_points * 2, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RepPoints head does not need
                this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (:obj:`ConfigDict`): Test / postprocessing configuration,
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
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list)
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, 2 * self.num_points)

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if bbox_pred.numel() != 0:
                # there exist the bug in cuda function `min_area_polygon` when
                # the input is small value. Transferring the predction of point
                # offsets to the real positions in the whole image can avoid
                # this issue sometimes. For more details, please refer to
                # https://github.com/open-mmlab/mmrotate/issues/405
                pts_pred = bbox_pred.reshape(-1, self.num_points, 2)
                pts_pred_offsety = pts_pred[:, :, 0::2]
                pts_pred_offsetx = pts_pred[:, :, 1::2]
                pts_pred = torch.cat([pts_pred_offsetx, pts_pred_offsety],
                                     dim=2).reshape(-1, 2 * self.num_points)
                pts_pos_center = priors[:, :2].repeat(1, self.num_points)
                pts = pts_pred * self.point_strides[level_idx] + pts_pos_center
                qboxes = min_area_polygons(pts)
                bboxes = qbox2rbox(qboxes)
            else:
                bboxes = bbox_pred.reshape((0, 5))

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.bboxes = RotatedBoxes(results.bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
