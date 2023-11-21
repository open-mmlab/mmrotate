# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import convex_iou


def points_center_pts(RPoints, y_first=True):
    """Compute center point of Pointsets.

    Args:
        RPoints (torch.Tensor): the  lists of Pointsets, shape (k, 18).
        y_first (bool, optional): if True, the sequence of Pointsets is (y,x).

    Returns:
        center_pts (torch.Tensor): the mean_center coordination of Pointsets,
            shape (k, 18).
    """
    RPoints = RPoints.reshape(-1, 9, 2)

    if y_first:
        pts_dy = RPoints[:, :, 0::2]
        pts_dx = RPoints[:, :, 1::2]
    else:
        pts_dx = RPoints[:, :, 0::2]
        pts_dy = RPoints[:, :, 1::2]
    pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
    pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
    center_pts = torch.cat([pts_dx_mean, pts_dy_mean], dim=1).reshape(-1, 2)
    return center_pts


def convex_overlaps(gt_bboxes, points):
    """Compute overlaps between polygons and points.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).
        points (torch.Tensor): Points to be assigned, shape(n, 18).

    Returns:
        overlaps (torch.Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
    """
    overlaps = convex_iou(points, gt_bboxes)
    overlaps = overlaps.transpose(1, 0)
    return overlaps


def levels_to_images(mlvl_tensor, flatten=False):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)
        flatten (bool, optional): if shape of mlvl_tensor is (N, C, H, W)
            set False, if shape of mlvl_tensor is  (N, H, W, C) set True.

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


def get_num_level_anchors_inside(num_level_anchors, inside_flags):
    """Get number of every level anchors inside.

    Args:
        num_level_anchors (List[int]): List of number of every level's anchors.
        inside_flags (torch.Tensor): Flags of all anchors.

    Returns:
        List[int]: List of number of inside anchors.
    """
    split_inside_flags = torch.split(inside_flags, num_level_anchors)
    num_level_anchors_inside = [
        int(flags.sum()) for flags in split_inside_flags
    ]
    return num_level_anchors_inside


def covariance_output_to_cholesky(pred_bbox_cov, num_factor=5):
    """
    Transforms output to covariance cholesky decomposition.
    Args:
        num_factor:
        pred_bbox_cov (kx4 or kx10): Output covariance matrix elements.

    Returns:
        predicted_cov_cholesky (kx4x4): cholesky factor matrix
    """
    # Embed diagonal variance
    diag_vars = torch.sqrt(torch.exp(pred_bbox_cov[:, 0:num_factor]))
    predicted_cov_cholesky = torch.diag_embed(diag_vars)

    if pred_bbox_cov.shape[1] > num_factor:
        tril_indices = torch.tril_indices(row=num_factor, col=num_factor, offset=-1)
        predicted_cov_cholesky[:, tril_indices[0],
        tril_indices[1]] = pred_bbox_cov[:, num_factor:]

    return predicted_cov_cholesky


def clamp_log_variance(pred_bbox_cov, clamp_min=-7.0, clamp_max=7.0):
    """
    Tiny function that clamps variance for consistency across all methods.
    """
    pred_bbox_var_component = torch.clamp(
        pred_bbox_cov[:, 0:4], clamp_min, clamp_max)
    return torch.cat((pred_bbox_var_component, pred_bbox_cov[:, 4:]), dim=1)


def get_probabilistic_loss_weight(current_step, annealing_step):
    """
    Tiny function to get adaptive probabilistic loss weight for consistency across all methods.
    """
    probabilistic_loss_weight = min(1.0, current_step / annealing_step)
    probabilistic_loss_weight = (
                                        100 ** probabilistic_loss_weight - 1.0) / (100.0 - 1.0)

    return probabilistic_loss_weight


def compute_mean_covariance_torch(input_samples):
    """
    Function for efficient computation of mean and covariance matrix in pytorch.

    Args:
        input_samples(list): list of tensors from M stochastic monte-carlo sampling runs, each containing N x k tensors.

    Returns:
        predicted_mean(Tensor): an Nxk tensor containing the predicted mean.
        predicted_covariance(Tensor): an Nxkxk tensor containing the predicted covariance matrix.

    """
    if isinstance(input_samples, torch.Tensor):
        num_samples = input_samples.shape[2]
    else:
        num_samples = len(input_samples)
        input_samples = torch.stack(input_samples, 2)

    # Compute Mean
    predicted_mean = torch.mean(input_samples, 2, keepdim=True)

    # Compute Covariance
    residuals = torch.transpose(
        torch.unsqueeze(
            input_samples -
            predicted_mean,
            1),
        1,
        3)
    predicted_covariance = torch.matmul(
        residuals, torch.transpose(residuals, 3, 2))
    predicted_covariance = torch.sum(
        predicted_covariance, 1) / (num_samples - 1)

    return predicted_mean.squeeze(2), predicted_covariance
