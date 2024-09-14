# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.logging import MMLogger
from torch import Tensor

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from .dota_metric import DOTAMetric


@METRICS.register_module()
class OHD_SJTUMetric(DOTAMetric):
    """OHD-SJTU evaluation metric, adding box head evaluation based on
    DOTAMetric.

    Note:  In addition to format the output results to JSON like CocoMetric,
    it can also generate the full image's results by merging patches' results.
    The premise is that you must use the tool provided by us to crop the
    OHD-SJTU large images, which can be found at: ``tools/data/dota/split``.

    Args:
    """

    default_prefix: Optional[str] = 'ohd-sjtu'

    def __init__(self,
                 *args,
                 metric: Union[str, List[str]] = 'mAP',
                 predict_box_type: str = 'rheadbox'):
        super().__init__(
            *args, metric=metric, predict_box_type=predict_box_type)

    def get_head_quadrant(self, rheadboxes: Tensor):
        """convert headxys in rheadboxes into head quadrants.

        Args:
            rheadboxes (Tensor): rotated head boxes, each of item is
                [x_c, y_c, w, h, t, head_x, head_y].

        Return:
            Tensor: each of item is [x_c, y_c, w, h, t, head_quadrant]
        """
        center_xys, head_xys = rheadboxes[..., :2], rheadboxes[..., -2:]
        original_shape = head_xys.shape[:-1]
        assert center_xys.size(-1) == 2 and head_xys.size(-1) == 2, \
            ('The last dimension of two input params must be 2, representing '
             f'xy coordinates, but got center_xys {center_xys.shape}, '
             f'head_xys {head_xys.shape}.')
        head_quadrants = []
        for center_xy, head_xy in zip(center_xys, head_xys):
            delta_x = head_xy[0] - center_xy[0]
            delta_y = head_xy[1] - center_xy[1]
            if (delta_x >= 0) and (delta_y >= 0):
                head_quadrants.append(0)
            elif (delta_x >= 0) and (delta_y <= 0):
                head_quadrants.append(1)
            elif (delta_x <= 0) and (delta_y <= 0):
                head_quadrants.append(2)
            else:
                head_quadrants.append(3)
        head_quadrants = head_xys.new_tensor(head_quadrants)
        head_quadrants = head_quadrants.view(*original_shape, 1)
        return torch.cat([rheadboxes[..., :5], head_quadrants], dim=-1)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            if gt_instances == {}:
                ann = dict()
            else:
                rheadbboxes = gt_instances['bboxes']
                rhead_ignore_bboxes = gt_ignore_instances['bboxes']
                # convert head xy in bboxes into head quadrant
                rbboxes_hqu = self.get_head_quadrant(rheadbboxes)
                if rhead_ignore_bboxes.shape[0] == 0:
                    rbboxes_hqu_ignore = rhead_ignore_bboxes[..., :6]
                else:
                    rbboxes_hqu_ignore = self.get_head_quadrant(
                        rhead_ignore_bboxes)
                ann = dict(
                    labels=gt_instances['labels'].cpu().numpy(),
                    bboxes=rbboxes_hqu.cpu().numpy(),
                    bboxes_ignore=rbboxes_hqu_ignore.cpu().numpy(),
                    labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['heads'] = pred['heads'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['pred_bbox_head_scores'] = []
            # get prediction in each class
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(result['labels'] == label)[0]
                pred_bbox_head_scores = np.hstack([
                    result['bboxes'][index],
                    result['heads'][index].reshape(-1, 1),
                    result['scores'][index].reshape(-1, 1)
                ])
                result['pred_bbox_head_scores'].append(pred_bbox_head_scores)

            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        # anns, results
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            # convert predictions to coco format and dump to json file
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_head_scores'] for pred in preds]

            mean_aps = []
            mean_heads = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, mean_head_acc, _ = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                mean_heads.append(mean_head_acc)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                eval_results[f'head_acc{int(iou_thr * 100):02d}'] = round(
                    mean_head_acc, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results['mean_head_acc'] = sum(mean_heads) / len(mean_heads)
            eval_results.move_to_end('mean_head_acc', last=False)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results
