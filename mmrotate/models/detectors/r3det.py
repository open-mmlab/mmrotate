# Copyright (c) SJTU. All rights reserved.
import torch.nn as nn

from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import FeatureRefineModule


@ROTATED_DETECTORS.register_module()
class R3Det(RotatedBaseDetector):
    """Rotated Refinement RetinaNet."""

    def __init__(self,
                 num_refine_stages,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 frm_cfgs=None,
                 refine_heads=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(R3Det, self).__init__()

        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.num_refine_stages = num_refine_stages
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            bbox_head.update(train_cfg=train_cfg['s0'])
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.init_weights()
        self.feat_refine_module = nn.ModuleList()
        self.refine_head = nn.ModuleList()
        for i, (frm_cfg,
                refine_head) in enumerate(zip(frm_cfgs, refine_heads)):
            self.feat_refine_module.append(FeatureRefineModule(**frm_cfg))
            if train_cfg is not None:
                refine_head.update(train_cfg=train_cfg['sr'][i])
            refine_head.update(test_cfg=test_cfg)
            self.refine_head.append(build_head(refine_head))
        for i in range(self.num_refine_stages):
            self.feat_refine_module[i].init_weights()
            self.refine_head[i].init_weights()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function."""
        losses = dict()
        x = self.extract_feat(img)

        outs = self.bbox_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f's0.{name}'] = value

        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            lw = self.train_cfg.stage_loss_weights[i]

            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss_refine = self.refine_head[i].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
            for name, value in loss_refine.items():
                losses[f'sr{i}.{name}'] = ([v * lw for v in value]
                                           if 'loss' in name else value)

            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.refine_head[-1].get_bboxes(*bbox_inputs, rois=rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels,
                         self.refine_head[-1].num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass
