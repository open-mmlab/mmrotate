# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

from mmdet.models.detectors.base import BaseDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import ModuleList
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class RefineSingleStageDetector(BaseDetector):
    """Base class for refine single-stage detectors, which used by `S2A-Net`
    and `R3Det`.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head_init (:obj:`ConfigDict` or dict): The bbox head module of
            the first stage.
        bbox_head_refine (list[:obj:`ConfigDict` | dict]): The bbox head
            module of the refine stage.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of RefineSingleStageDetector. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of RefineSingleStageDetector. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head_init: OptConfigType = None,
                 bbox_head_refine: List[OptConfigType] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if train_cfg is not None:
            bbox_head_init.update(train_cfg=train_cfg['init'])
        bbox_head_init.update(test_cfg=test_cfg)
        self.bbox_head_init = MODELS.build(bbox_head_init)
        self.num_refine_stages = len(bbox_head_refine)
        self.bbox_head_refine = ModuleList()
        for i, refine_head in enumerate(bbox_head_refine):
            if train_cfg is not None:
                refine_head.update(train_cfg=train_cfg['refine'][i])
            refine_head.update(test_cfg=test_cfg)
            self.bbox_head_refine.append(MODELS.build(refine_head))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()
        outs = self.bbox_head_init(x)
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        init_losses = self.bbox_head_init.loss_by_feat(*loss_inputs)
        keys = init_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'init' not in key:
                init_losses[f'{key}_init'] = init_losses.pop(key)
        losses.update(init_losses)

        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            weight = self.train_cfg.stage_loss_weights[i]
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                                  batch_gt_instances_ignore)
            refine_losses = self.bbox_head_refine[i].loss_by_feat(
                *loss_inputs, rois=rois)
            keys = refine_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'refine' not in key:
                    loss = refine_losses.pop(key)
                    if isinstance(loss, Sequence):
                        loss = [item * weight for item in loss]
                    else:
                        loss = loss * weight
                    refine_losses[f'{key}_refine_{i}'] = loss
            losses.update(refine_losses)

            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois=rois)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 5),
              the last dimension 5 arrange as (x, y, w, h, t).
        """
        x = self.extract_feat(batch_inputs)
        outs = self.bbox_head_init(x)
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        predictions = self.bbox_head_refine[-1].predict_by_feat(
            *outs, rois=rois, batch_img_metas=batch_img_metas, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, predictions)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        outs = self.bbox_head_init(x)
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois)

        return outs

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
