# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor

from mmrotate.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils import unpack_gt_instances

@MODELS.register_module()
class RefineSingleStageDetector(SingleStageDetector):
    """Base class for refine single-stage detectors, which used by `S2A-Net`
    and `R3Det`.

    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head_init: OptConfigType = None,
                 bbox_head_refine: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head_init.update(train_cfg=train_cfg)
        bbox_head_init.update(test_cfg=test_cfg)
        bbox_head_refine.update(train_cfg=train_cfg)
        bbox_head_refine.update(test_cfg=test_cfg)
        self.bbox_head_init = MODELS.build(bbox_head_init)
        self.bbox_head_refine = MODELS.build(bbox_head_refine)
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
 
        for name, value in init_losses.items():
            losses[f'init_{name}'] = value
        rois = self.bbox_head_refine[i].filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            lw = self.train_cfg.stage_loss_weights[i]
            x_refine = self.bbox_head_refine[i](x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                                batch_gt_instances_ignore)
            refine_losses = self.bbox_head_refine[i].loss(*loss_inputs, rois=rois)
            for name, value in refine_losses.items():
                losses[f'refine{i}_{name}'] = ([v * lw for v in value]
                                           if 'loss' in name else value)
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
        rois = self.bbox_head_refine[0].filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            x_refine = self.bbox_head_refine[i](x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        predictions = self.predict_by_feat(
            *outs, rois, batch_img_metas=batch_img_metas, rescale=rescale)

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
        rois = self.bbox_head_refine[0].filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            x_refine = self.bbox_head_refine[i](x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois)
                
        return outs

