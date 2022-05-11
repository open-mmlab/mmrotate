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
            self.prior_generator = build_prior_generator(prior_generator)
            assert len(align_modules) == len(heads)
            self.pre_align = True
        else:
            self.prior_generator = None
            assert len(align_modules) == len(heads) - 1
            align_modules.insert(0, dict(type='PseudoAlignModule'))
            self.pre_align = False

        # Get Stage Number
        self.num_stages = len(heads)
        # Build align_modules
        align_modules = [
            build_align_module(align_module) for align_module in align_modules
        ]
        self.align_modules = nn.ModuleList(align_modules)

        # Process train_cfg
        if train_cfg:
            if isinstance(train_cfg, dict):
                self.train_cfg = [
                    deepcopy(train_cfg) for _ in range(self.num_stages)
                ]
            else:
                assert len(train_cfg) == self.num_stages
                self.train_cfg = train_cfg
        else:
            self.train_cfg = [None for _ in range(self.num_stages)]

        # Process test_cfg
        if isinstance(test_cfg, dict):
            self.test_cfg = [
                deepcopy(test_cfg) for _ in range(self.num_stages)
            ]
        else:
            assert len(test_cfg) == self.num_stages
            self.test_cfg = test_cfg

        # Build heads
        self.heads = nn.ModuleList()
        for i, head in enumerate(heads):
            head.update(train_cfg=self.train_cfg[i])
            head.update(test_cfg=self.test_cfg[i])
            self.heads.append(build_head(head))

    def loss(self, **kwargs):
        raise NotImplementedError

    def get_init_anchors(self, featmap_sizes, num_imgs, device='cuda'):
        """Get init anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            num_imgs (int): Numbers of image.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple (list[Tensor]):

                - anchor_list (list[Tensor]): Anchors of each image.
        """

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        return anchor_list

    def forward(self, feats):
        if self.pre_align:
            featmap_sizes = [featmap.size()[-2:] for featmap in feats]
            num_imgs = feats[0].size(0)
            rois = self.get_init_anchors(featmap_sizes, num_imgs)
        else:
            rois = None
        outs = None
        for i in range(self.num_stages):
            feats = self.align_modules[i](feats, rois)
            outs = self.heads[i](feats)
            if i == 0 and not self.pre_align:
                rois = self.heads[i].refine_bboxes(*outs)
            elif i != self.num_stages - 1:
                rois = self.heads[i].refine_bboxes(*outs, rois=rois)
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

        if self.pre_align:
            featmap_sizes = [featmap.size()[-2:] for featmap in x]
            num_imgs = x[0].size(0)
            rois = self.get_init_anchors(featmap_sizes, num_imgs)
        else:
            rois = None

        for i in range(self.num_stages):
            align_feat = self.align_modules[i](x, rois)
            outs, loss = self._bbox_forward_train(
                i,
                align_feat,
                gt_bboxes,
                gt_labels,
                img_metas,
                gt_bboxes_ignore,
                rois=rois)

            if i == 0 and not self.pre_align:
                rois = self.heads[i].refine_bboxes(*outs)
            elif i != self.num_stages - 1:
                rois = self.heads[i].refine_bboxes(*outs, rois=rois)

            # update loss
            for name, value in loss.items():
                losses[f's{i}.{name}'] = value

        return losses

    def get_bboxes(self, *args, **kwargs):
        return self.heads[-1].get_bboxes(*args, **kwargs)
