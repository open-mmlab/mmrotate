# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.futures
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

from mmrotate.models.detectors.semi_base import RotatedSemiBaseDetector
from mmrotate.models.losses import OT_Loss
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox2qbox, rbox_project


@MODELS.register_module()
class SOOD(RotatedSemiBaseDetector):
    """Implementation of `SOOD: Towards Semi-Supervised Oriented Object
    Detection <https://arxiv.org/abs/2304.04515>`

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_loss_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised loss config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_loss_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # loss settings
        self.semi_loss_cfg = semi_loss_cfg
        self.cls_pseudo_thr = self.semi_loss_cfg.get('cls_pseudo_thr', 0.5)
        self.cls_loss_type = self.semi_loss_cfg.get('cls_loss_type', 'BCE')

        self.reg_loss_type = self.semi_loss_cfg.get('reg_loss_type',
                                                    'SmoothL1Loss')
        assert self.reg_loss_type in ['SmoothL1Loss']
        self.loss_bbox = nn.SmoothL1Loss(reduction='none')

        # aux loss settings
        self.aux_loss = self.semi_loss_cfg.get('aux_loss', None)
        if self.aux_loss is not None:
            assert self.aux_loss in ['ot_loss_norm', 'ot_ang_loss_norm']
            self.aux_loss_cfg = self.semi_loss_cfg.get('aux_loss_cfg', None)
            assert self.aux_loss_cfg is not None, \
                'aux_loss_cfg should be provided when aux_loss is not None.'
            self.ot_weight = self.aux_loss_cfg.pop('loss_weight', 1.)
            self.cost_type = self.aux_loss_cfg.pop('cost_type', 'all')
            assert self.cost_type in ['all', 'dist', 'score']
            self.clamp_ot = self.aux_loss_cfg.pop('clamp_ot', False)
            self.gc_loss = OT_Loss(**self.aux_loss_cfg)

        self.rbox_pts_ratio = self.semi_loss_cfg.get('rbox_pts_ratio', 0.25)
        self.dynamic_weight = self.semi_loss_cfg.get('dynamic_weight', '50ang')
        assert self.dynamic_weight in [
            'None', 'ang', '10ang', '50ang', '100ang'
        ]

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        dense_predicts = self.teacher(batch_inputs)
        batch_info = {}
        batch_info['dense_predicts'] = dense_predicts

        self.teacher.eval()
        results_list = self.teacher.predict(batch_inputs, batch_data_samples)

        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = copy.deepcopy(results.pred_instances)
            data_samples.gt_instances.bboxes = rbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        return batch_data_samples, batch_info

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.
        Returns:
            dict: A dictionary of loss components
        """

        gpu_device = batch_inputs.device
        # first filter the pseudo instances with cls scores
        batch_data_samples = filter_gt_instances(
            batch_data_samples, score_thr=self.cls_pseudo_thr)

        # decide the dense pseudo label area
        # according to the teacher predictions
        masks = torch.zeros(
            len(batch_data_samples),
            batch_inputs.shape[-2],
            batch_inputs.shape[-1],
            device=gpu_device)
        for img_idx, data_samples in enumerate(batch_data_samples):
            if len(data_samples.gt_instances) > 0:
                qboxes = rbox2qbox(data_samples.gt_instances.bboxes)
                # different with the original implementation,
                # we use round not just int()
                # pts = qboxes.cpu().numpy().astype(int)
                pts = np.round(qboxes.cpu().numpy()).astype(int)
                a = []
                for i in range(len(pts)):
                    a.append(np.split(pts[i], 4))
                pts = np.array(a)
                mask = np.zeros(
                    (batch_inputs.shape[-2], batch_inputs.shape[-1]),
                    dtype=np.uint8)
                cv2.fillPoly(mask, pts, 255)
                valid_mask = np.zeros_like(mask)
                valid_pts = np.transpose(mask.nonzero())
                select_list = list(range(len(valid_pts)))
                for _ in range(3):
                    random.shuffle(select_list)
                select_num = int(self.rbox_pts_ratio * len(valid_pts))
                valid_pts = valid_pts[select_list[:select_num]]
                valid_mask[valid_pts[:, 0], valid_pts[:, 1]] = 255
                masks[img_idx] = torch.from_numpy(valid_mask) > 0

        masks = masks.view(-1, 1, batch_inputs.shape[-2],
                           batch_inputs.shape[-1])

        # interpolate the dense pseudo label area to FPN P5
        size_fpn_p5 = (int(batch_inputs.shape[-2] / 8),
                       int(batch_inputs.shape[-1] / 8))

        masks = F.interpolate(
            masks.float(), size=size_fpn_p5).bool().squeeze(1)

        num_valid = sum([_.sum() for _ in masks]) if isinstance(
            masks, list) else masks.sum()

        if num_valid == 0:
            loss_cls = torch.tensor(0., device=gpu_device)
            loss_bbox = torch.tensor(0., device=gpu_device)
            loss_centerness = torch.tensor(0., device=gpu_device)
            if self.aux_loss is not None:
                loss_gc = torch.tensor(0., device=gpu_device)
                losses = {
                    'loss_cls': loss_cls,
                    'loss_bbox': loss_bbox,
                    'loss_centerness': loss_centerness,
                    'loss_gc': loss_gc
                }
            else:
                losses = {
                    'loss_cls': loss_cls,
                    'loss_bbox': loss_bbox,
                    'loss_centerness': loss_centerness,
                }
            return rename_loss_dict('unsup_', losses)
        else:
            teacher_logit = batch_info['dense_predicts']
            teacher_cls_scores_logits, teacher_bbox_preds, \
                teacher_angle_pred, teacher_centernesses = teacher_logit

            student_logit = self.student(batch_inputs)
            student_cls_scores_logits, student_bbox_preds, \
                student_angle_pred, student_centernesses = student_logit

            loss_cls = torch.tensor(0., device=gpu_device)
            loss_bbox = torch.tensor(0., device=gpu_device)
            loss_centerness = torch.tensor(0., device=gpu_device)
            for i in range(len(masks)):
                if masks[i].sum() == 0:
                    continue
                teacher_cls_scores_logits_ = (
                    teacher_cls_scores_logits[0][i]).permute(1, 2, 0)[masks[i]]
                teacher_bbox_preds_ = (teacher_bbox_preds[0][i]).permute(
                    1, 2, 0)[masks[i]]
                teacher_angle_pred_ = (teacher_angle_pred[0][i]).permute(
                    1, 2, 0)[masks[i]]
                teacher_centernesses_ = (teacher_centernesses[0][i]).permute(
                    1, 2, 0)[masks[i]]

                student_cls_scores_logits_ = (
                    student_cls_scores_logits[0][i]).permute(1, 2, 0)[masks[i]]
                student_bbox_preds_ = (student_bbox_preds[0][i]).permute(
                    1, 2, 0)[masks[i]]
                student_angle_pred_ = (student_angle_pred[0][i]).permute(
                    1, 2, 0)[masks[i]]
                student_centernesses_ = (student_centernesses[0][i]).permute(
                    1, 2, 0)[masks[i]]

                teacher_bbox_preds_ = torch.cat(
                    [teacher_bbox_preds_, teacher_angle_pred_], dim=-1)
                student_bbox_preds_ = torch.cat(
                    [student_bbox_preds_, student_angle_pred_], dim=-1)

                with torch.no_grad():
                    if self.dynamic_weight in [
                            'None', 'ang', '10ang', '50ang', '100ang'
                    ]:
                        loss_weight = torch.abs(
                            teacher_bbox_preds_[:, -1] -
                            student_bbox_preds_[:, -1]) / np.pi
                        if self.dynamic_weight == 'None':
                            loss_weight = torch.ones_like(
                                loss_weight.unsqueeze(-1))
                        elif self.dynamic_weight == 'ang':
                            loss_weight = torch.clamp(
                                loss_weight.unsqueeze(-1), 0, 1) + 1
                        elif self.dynamic_weight == '10ang':
                            loss_weight = torch.clamp(
                                10 * loss_weight.unsqueeze(-1), 0, 1) + 1
                        elif self.dynamic_weight == '50ang':
                            loss_weight = torch.clamp(
                                50 * loss_weight.unsqueeze(-1), 0, 1) + 1
                        elif self.dynamic_weight == '100ang':
                            loss_weight = torch.clamp(
                                100 * loss_weight.unsqueeze(-1), 0, 1) + 1
                        else:
                            loss_weight = loss_weight.unsqueeze(-1) + 1
                    else:
                        raise NotImplementedError

                # cls loss
                if self.cls_loss_type == 'BCE':
                    loss_cls_ = F.binary_cross_entropy(
                        student_cls_scores_logits_.sigmoid(),
                        teacher_cls_scores_logits_.sigmoid(),
                        reduction='none')
                else:
                    raise NotImplementedError
                loss_cls_ = (loss_cls_ * loss_weight).mean()
                # bbox loss
                loss_bbox_ = self.loss_bbox(
                    student_bbox_preds_,
                    teacher_bbox_preds_) * teacher_centernesses_.sigmoid()
                loss_bbox_ = (loss_bbox_ * loss_weight).mean()

                # centerness loss
                loss_centerness_ = F.binary_cross_entropy(
                    student_centernesses_.sigmoid(),
                    teacher_centernesses_.sigmoid(),
                    reduction='none')
                loss_centerness_ = (loss_centerness_ * loss_weight).mean()

                loss_cls += loss_cls_
                loss_bbox += loss_bbox_
                loss_centerness += loss_centerness_

            loss_cls = loss_cls / len(masks)
            loss_bbox = loss_bbox / len(masks)
            loss_centerness = loss_centerness / len(masks)

        if self.aux_loss is not None:
            loss_gc = torch.zeros(1, device=gpu_device)
            if self.aux_loss == 'ot_ang_loss_norm':
                teacher_score_map = teacher_logit[2][0]
                student_score_map = student_logit[2][0]
            elif self.aux_loss == 'ot_loss_norm':
                teacher_score_map = teacher_logit[0][0]
                student_score_map = student_logit[0][0]

            teacher_score_map = teacher_score_map.permute(0, 2, 3, 1)
            student_score_map = student_score_map.permute(0, 2, 3, 1)

            if self.aux_loss == 'ot_ang_loss_norm':
                teacher_score_map = torch.abs(teacher_score_map) / np.pi
                student_score_map = torch.abs(student_score_map) / np.pi
            elif self.aux_loss == 'ot_loss_norm':
                teacher_score_map = torch.softmax(teacher_score_map, dim=-1)
                student_score_map = torch.softmax(student_score_map, dim=-1)

            for i in range(teacher_score_map.shape[0]):
                teacher_score, score_cls = torch.max(
                    teacher_score_map[i][masks[i]], dim=-1)
                student_score = student_score_map[i][masks[i]][
                    torch.arange(teacher_score.shape[0]), score_cls]
                pts = masks[i].nonzero()
                if len(pts) <= 1:
                    continue
                loss_gc += self.gc_loss(
                    teacher_score,
                    student_score,
                    pts,
                    cost_type=self.cost_type,
                    clamp_ot=self.clamp_ot)
            loss_gc = self.ot_weight * loss_gc / len(teacher_score_map)

        if self.aux_loss is not None:
            losses = {
                'loss_cls': loss_cls,
                'loss_bbox': loss_bbox,
                'loss_centerness': loss_centerness,
                'loss_gc': loss_gc
            }
        else:
            losses = {
                'loss_cls': loss_cls,
                'loss_bbox': loss_bbox,
                'loss_centerness': loss_centerness,
            }

        # apply burnin strategy to reweight the unsupervised weights
        burn_in_steps = self.semi_train_cfg.get('burn_in_steps', 5000)
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        self.weight_suppress = self.semi_train_cfg.get('weight_suppress',
                                                       'linear')
        if self.weight_suppress == 'exp':
            target = burn_in_steps + 2000
            if self.iter_count <= target:
                scale = np.exp((self.iter_count - target) / 1000)
                unsup_weight *= scale
        elif self.weight_suppress == 'step':
            target = burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= 0.25
        elif self.weight_suppress == 'linear':
            target = burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= (self.iter_count -
                                 burn_in_steps) / burn_in_steps
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))
