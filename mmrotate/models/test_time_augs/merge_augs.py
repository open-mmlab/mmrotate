# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch

from mmrotate.structures.bbox import rbbox_mapping_back


# TODO remove this in boxlist
def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.
    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        bboxes = rbbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                    flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


# TODO remove this in boxlist
def merge_aug_results(aug_batch_results, aug_batch_img_metas):
    """Merge augmented detection results, only bboxes corresponding score under
    flipping and multi-scale resizing can be processed now.
    Args:
        aug_batch_results (list[list[[obj:`InstanceData`]]):
            Detection results of multiple images with
            different augmentations.
            The outer list indicate the augmentation . The inter
            list indicate the batch dimension.
            Each item usually contains the following keys.
            - scores (Tensor): Classification scores, in shape
              (num_instance,)
            - labels (Tensor): Labels of bboxes, in shape
              (num_instances,).
            - bboxes (Tensor): In shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        aug_batch_img_metas (list[list[dict]]): The outer list
            indicates test-time augs (multiscale, flip, etc.)
            and the inner list indicates
            images in a batch. Each dict in the list contains
            information of an image in the batch.
    Returns:
        batch_results (list[obj:`InstanceData`]): Same with
        the input `aug_results` except that all bboxes have
        been mapped to the original scale.
    """
    num_augs = len(aug_batch_results)
    num_imgs = len(aug_batch_results[0])

    batch_results = []
    aug_batch_results = copy.deepcopy(aug_batch_results)
    for img_id in range(num_imgs):
        aug_results = []
        for aug_id in range(num_augs):
            img_metas = aug_batch_img_metas[aug_id][img_id]
            results = aug_batch_results[aug_id][img_id]

            img_shape = img_metas['img_shape']
            scale_factor = img_metas['scale_factor']
            flip = img_metas['flip']
            flip_direction = img_metas['flip_direction']
            bboxes = rbbox_mapping_back(results.bboxes, img_shape,
                                        scale_factor, flip, flip_direction)
            results.bboxes = bboxes
            aug_results.append(results)
        merged_aug_results = results.cat(aug_results)
        batch_results.append(merged_aug_results)

    return batch_results
