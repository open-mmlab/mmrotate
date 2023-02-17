# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    elif angle_range == 'r360':
        return (angle + np.pi) % (2 * np.pi) - np.pi
    else:
        print('Not yet implemented.')


def gaussian2bbox(gmm):
    """Convert Gaussian distribution to polygons by SVD.

    Args:
        gmm (dict[str, torch.Tensor]): Dict of Gaussian distribution.

    Returns:
        torch.Tensor: Polygons.
    """
    try:
        from torch_batch_svd import svd
    except ImportError:
        svd = None
    L = 3
    var = gmm.var
    mu = gmm.mu
    assert mu.size()[1:] == (1, 2)
    assert var.size()[1:] == (1, 2, 2)
    T = mu.size()[0]
    var = var.squeeze(1)
    if svd is None:
        raise ImportError('Please install torch_batch_svd first.')
    U, s, Vt = svd(var)
    size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)
    mu = mu.repeat(1, 4, 1)
    dx_dy = size_half * torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]],
                                     dtype=torch.float32,
                                     device=size_half.device)
    bboxes = (mu + dx_dy.matmul(Vt.transpose(1, 2))).reshape(T, 8)

    return bboxes


def gt2gaussian(target):
    """Convert polygons to Gaussian distributions.

    Args:
        target (torch.Tensor): Polygons with shape (N, 8).

    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    L = 3
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))


def distance2obb(points: torch.Tensor,
                 distance: torch.Tensor,
                 angle_version: str = 'oc'):
    """Convert distance angle to rotated boxes.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries and angle (left, top, right, bottom, angle).
            Shape (B, N, 5) or (N, 5)
        angle_version: angle representations.
    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    distance, angle = distance.split([4, 1], dim=-1)

    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)

    rot_matrix = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle],
                           dim=-1)
    rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)

    wh = distance[..., :2] + distance[..., 2:]
    offset_t = (distance[..., 2:] - distance[..., :2]) / 2
    offset_t = offset_t.unsqueeze(-1)
    offset = torch.matmul(rot_matrix, offset_t).squeeze(-1)
    ctr = points[..., :2] + offset

    angle_regular = norm_angle(angle, angle_version)
    return torch.cat([ctr, wh, angle_regular], dim=-1)
