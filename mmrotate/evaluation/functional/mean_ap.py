# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_quadri, box_iou_rotated
from mmdet.evaluation.functional import average_precision
from mmengine.logging import print_log
from terminaltables import AsciiTable


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 box_type='rbox',
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6)
            or (m, 7).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5)
            or (n, 6).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5) or (k, 6). Defaults to None
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        box_type (str): Box type. If the QuadriBoxes or RotatedHeadBoxes is
            used, you need to specify 'qbox' or 'rheadbox'. Defaults to
            'rbox'.
        area_ranges (list[tuple], optional): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Defaults to None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    # an indicator of ignored gts
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    tp_head = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        if box_type == 'rheadbox':
            return tp, fp, tp_head
        else:
            return tp, fp

    if box_type == 'rbox':
        ious = box_iou_rotated(
            torch.from_numpy(det_bboxes).float(),
            torch.from_numpy(gt_bboxes).float()).numpy()
    elif box_type == 'rheadbox':
        ious = box_iou_rotated(
            torch.from_numpy(det_bboxes[..., :5]).float(),
            torch.from_numpy(gt_bboxes[..., :5]).float()).numpy()
    elif box_type == 'qbox':
        ious = box_iou_quadri(
            torch.from_numpy(det_bboxes).float(),
            torch.from_numpy(gt_bboxes).float()).numpy()
    else:
        raise NotImplementedError
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        if box_type == 'rheadbox':
                            if gt_bboxes[matched_gt, -1] == det_bboxes[i, -2]:
                                tp[k, i] = 1
                                tp_head[k, i] = 1
                        else:
                            tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                if box_type == 'rbox':
                    bbox = det_bboxes[i, :5]
                    area = bbox[2] * bbox[3]
                elif box_type == 'rheadbox':
                    bbox = det_bboxes[i, :5]
                    area = bbox[2] * bbox[3]
                elif box_type == 'qbox':
                    bbox = det_bboxes[i, :8]
                    pts = bbox.reshape(*bbox.shape[:-1], 4, 2)
                    roll_pts = torch.roll(pts, 1, dims=-2)
                    xyxy = torch.sum(
                        pts[..., 0] * roll_pts[..., 1] -
                        roll_pts[..., 0] * pts[..., 1],
                        dim=-1)
                    area = 0.5 * torch.abs(xyxy)
                else:
                    raise NotImplementedError
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    if box_type == 'rheadbox':
        return tp, fp, tp_head
    else:
        return tp, fp


def get_cls_results(det_results, annotations, class_id, box_type):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.
        box_type (str): Box type. If the QuadriBoxes is used, you need to
            specify 'qbox'. Defaults to 'rbox'.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        if len(ann['bboxes']) != 0:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            if box_type == 'rbox':
                cls_gts.append(torch.zeros((0, 5), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))
            elif box_type == 'qbox':
                cls_gts.append(torch.zeros((0, 8), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 8), dtype=torch.float64))
            elif box_type == 'rheadbox':
                cls_gts.append(torch.zeros((0, 6), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 6)))
            else:
                raise NotImplementedError

    return cls_dets, cls_gts, cls_gts_ignore


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   box_type='rbox',
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5) or (n, 6)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
                or (k, 6)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple], optional): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Defaults to None.
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        box_type (str): Box type. If the QuadriBoxes is used, you need to
            specify 'qbox'. Defaults to 'rbox'.
        dataset (list[str] | str, optional): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Defaults to None.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details.
            Defaults to None.
        nproc (int): Processes used for computing TP and FP.
            Defaults to 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i, box_type)

        # compute tp and fp for each image with multiple processes
        if box_type == 'rheadbox':
            tpfphead = pool.starmap(
                tpfp_default,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou_thr for _ in range(num_imgs)],
                    [box_type for _ in range(num_imgs)],
                    [area_ranges for _ in range(num_imgs)]))
            tp, fp, tp_head = tuple(zip(*tpfphead))
        else:
            tpfp = pool.starmap(
                tpfp_default,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou_thr for _ in range(num_imgs)],
                    [box_type for _ in range(num_imgs)],
                    [area_ranges for _ in range(num_imgs)]))
            tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                if box_type == 'rbox' or box_type == 'rheadbox':
                    gt_areas = bbox[:, 2] * bbox[:, 3]
                elif box_type == 'qbox':
                    pts = bbox.reshape(*bbox.shape[:-1], 4, 2)
                    roll_pts = torch.roll(pts, 1, dims=-2)
                    xyxy = torch.sum(
                        pts[..., 0] * roll_pts[..., 1] -
                        roll_pts[..., 0] * pts[..., 1],
                        dim=-1)
                    gt_areas = 0.5 * torch.abs(xyxy)
                else:
                    raise NotImplementedError
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
        if box_type == 'rheadbox':
            tp_head = np.hstack(tp_head)
            head_acc = []
            for tp_s_head in tp_head:
                head_acc.append(sum(tp_s_head) / len(tp_s_head))
            head_acc = np.array(head_acc)
            eval_results[-1]['head_acc'] = head_acc
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    if box_type == 'rheadbox':
        if scale_ranges is not None:
            all_head_acc = np.vstack(
                [cls_result['head_acc'] for cls_result in eval_results])
            mean_head_acc = []
            for i in range(num_scales):
                if np.any(all_num_gts[:, i] > 0):
                    mean_head_acc.append(all_head_acc[all_num_gts[:, i] > 0,
                                                      i].mean())
                else:
                    mean_head_acc.append(0.0)
        else:
            head_accs = []
            for cls_result in eval_results:
                if cls_result['num_gts'] > 0:
                    head_accs.append(cls_result['head_acc'])
            mean_head_acc = np.array(
                head_accs).mean().item() if head_accs else 0.0
        print_map_summary(
            mean_ap,
            eval_results,
            dataset,
            area_ranges,
            mean_head_acc=mean_head_acc,
            logger=logger)
        return mean_ap, mean_head_acc, eval_results
    else:
        print_map_summary(
            mean_ap, eval_results, dataset, area_ranges, logger=logger)
        return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      mean_head_acc=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str, optional): Dataset name or dataset classes.
        scale_ranges (list[tuple], optional): Range of scales to be evaluated.
        mean_head_acc (float): Calculated from 'eval_map()', mean head
            detection accuracy, if None, no head detection.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details.
            Defaults to None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    head_accs = np.zeros((num_scales, num_classes), dtype=np.float32)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']
        head_accs[:, i] = cls_result['head_acc'] if mean_head_acc else None

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]
    if mean_head_acc and not isinstance(mean_head_acc, list):
        mean_head_acc = [mean_head_acc]

    header = ['class', 'gts', 'dets', 'recall', 'ap', 'head_acc'] if \
        mean_head_acc else ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            if mean_head_acc:
                row_data = [
                    label_names[j], num_gts[i, j], results[j]['num_dets'],
                    f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}',
                    f'{head_accs[i, j]:.3f}'
                ]
            else:
                row_data = [
                    label_names[j], num_gts[i, j], results[j]['num_dets'],
                    f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
                ]
            table_data.append(row_data)
        if mean_head_acc:
            table_data.append([
                'mAP', '', '', '', f'{mean_ap[i]:.3f}',
                f'{mean_head_acc[i]:.3f}'
            ])
        else:
            table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
