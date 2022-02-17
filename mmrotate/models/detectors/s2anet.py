# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import AlignConvModule


@ROTATED_DETECTORS.register_module()
class S2ANet(RotatedBaseDetector):
    """Rotated Refinement RetinaNet."""

    def __init__(self,
                 backbone,
                 neck=None,
                 fam_head=None,
                 align_cfgs=None,
                 odm_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(S2ANet, self).__init__()

        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            fam_head.update(train_cfg=train_cfg['fam_cfg'])
        fam_head.update(test_cfg=test_cfg)
        self.fam_head = build_head(fam_head)
        self.fam_head.init_weights()

        self.align_conv_type = align_cfgs['type']
        self.align_conv_size = align_cfgs['kernel_size']
        self.feat_channels = align_cfgs['channels']
        self.featmap_strides = align_cfgs['featmap_strides']

        if self.align_conv_type == 'AlignConv':
            self.align_conv = AlignConvModule(self.feat_channels,
                                              self.featmap_strides,
                                              self.align_conv_size)

        if train_cfg is not None:
            odm_head.update(train_cfg=train_cfg['odm_cfg'])
        odm_head.update(test_cfg=test_cfg)
        self.odm_head = build_head(odm_head)
        self.odm_head.init_weights()

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
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)

        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function of S2ANet."""
        losses = dict()
        x = self.extract_feat(img)

        outs = self.fam_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.fam_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f'fam.{name}'] = value

        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_refine = self.odm_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
        for name, value in loss_refine.items():
            losses[f'odm.{name}'] = value

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
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.odm_head.get_bboxes(*bbox_inputs, rois=rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.odm_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
