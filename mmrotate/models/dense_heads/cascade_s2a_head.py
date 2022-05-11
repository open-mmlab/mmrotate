# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch.nn as nn
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmrotate.core import build_prior_generator
from ..builder import ROTATED_HEADS, build_head
from ..utils.align_module import build_align_module


@ROTATED_HEADS.register_module()
class CascadeS2AHead(BaseDenseHead):

    def __init__(self,
                 prior_generator=None,
                 align_modules=None,
                 heads=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CascadeS2AHead, self).__init__(init_cfg)
        if prior_generator is not None:
            self.anchor_generator = build_prior_generator(prior_generator)
            assert len(align_modules) == len(heads)
            self.pre_align = True
        else:
            self.anchor_generator = None
            assert len(align_modules) == len(heads) - 1
            self.pre_align = False

        # TODO 支持非数组
        assert train_cfg is None or len(train_cfg) == len(heads)

        self.num_stages = len(heads)

        align_modules = [
            build_align_module(align_module) for align_module in align_modules
        ]
        self.align_modules = nn.ModuleList(align_modules)

        self.heads = nn.ModuleList()
        for i, head in enumerate(heads):
            if train_cfg is not None:
                head.update(train_cfg=train_cfg[i])
            head.update(test_cfg=deepcopy(test_cfg))
            self.heads.append(build_head(head))

    def loss(self, **kwargs):
        raise NotImplementedError

    def forward(self, feats):
        rois = None
        outs = None
        for i in range(self.num_stages):
            outs = self.heads[i](feats)
            if i != (self.num_stages - 1):
                rois = self.heads[i].refine_bboxes(*outs)
                feats = self.align_modules[i](feats, rois)
        return outs, rois

    def _bbox_forward_train(self,
                            stage,
                            x,
                            gt_bboxes,
                            gt_labels,
                            img_metas,
                            gt_bboxes_ignore=None,
                            rois=None):
        """Run forward function and calculate loss for box head in training."""
        outs = self.heads[stage](x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        if rois is None:
            losses = self.heads[stage].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        else:
            losses = self.heads[stage].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)

        return outs, losses

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        losses = dict()

        rois = None
        align_feat = x

        for i in range(self.num_stages):

            if i == 0 and self.pre_align:
                # TODO
                rois = None
                align_feat = self.align_modules[0](x, rois)

            outs, loss = self._bbox_forward_train(
                i,
                align_feat,
                gt_bboxes,
                gt_labels,
                img_metas,
                gt_bboxes_ignore,
                rois=rois)

            if i != self.num_stages - 1:
                # TODO add rois
                rois = self.heads[i].refine_bboxes(*outs)

                # TODO ERROR i
                align_feat = self.align_modules[i](x, rois)
            # update loss
            for name, value in loss.items():
                losses[f's{i}.{name}'] = value

        return losses

    def get_bboxes(self, *args, **kwargs):
        return self.heads[-1].get_bboxes(*args, **kwargs)
