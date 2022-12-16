# Copyright (c) SJTU. All rights reserved.
from copy import deepcopy

import torch
from torch import nn

from mmrotate.registry import MODELS


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def gwd_loss(pred, target, fun='sqrt', tau=2.0):
    """Gaussian Wasserstein distance loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    xy_distance = (mu_p - mu_t).square().sum(dim=-1)

    whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (sigma_p.bmm(sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (sigma_p.det() * sigma_t.det()).clamp(0).sqrt()
    whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()

    dis = xy_distance + whr_distance
    gwd_dis = dis.clamp(min=1e-6)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + torch.sqrt(gwd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + torch.log1p(gwd_dis))
    else:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
        loss = torch.log1p(torch.sqrt(gwd_dis) / scale)
    return loss


def bcd_loss(pred, target, fun='log1p', tau=1.0):
    """Bhatacharyya distance loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1, 2)
    mu_t = mu_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    delta = (mu_p - mu_t).unsqueeze(-1)
    sigma = 0.5 * (sigma_p + sigma_t)
    sigma_inv = torch.inverse(sigma)

    term1 = torch.log(
        torch.det(sigma) /
        (torch.sqrt(torch.det(sigma_t.matmul(sigma_p))))).reshape(-1, 1)
    term2 = delta.transpose(-1, -2).matmul(sigma_inv).matmul(delta).squeeze(-1)
    dis = 0.5 * term1 + 0.125 * term2
    bcd_dis = dis.clamp(min=1e-6)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + torch.sqrt(bcd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + torch.log1p(bcd_dis))
    else:
        loss = 1 - 1 / (tau + bcd_dis)
    return loss


def kld_loss(pred, target, fun='log1p', tau=1.0):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1, 2)
    mu_t = mu_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    delta = (mu_p - mu_t).unsqueeze(-1)
    sigma_t_inv = torch.inverse(sigma_t)
    term1 = delta.transpose(-1,
                            -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)
    term2 = torch.diagonal(
        sigma_t_inv.matmul(sigma_p),
        dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) + \
        torch.log(torch.det(sigma_t) / torch.det(sigma_p)).reshape(-1, 1)
    dis = term1 + term2 - 2
    kl_dis = dis.clamp(min=1e-6)

    if fun == 'sqrt':
        kl_loss = 1 - 1 / (tau + torch.sqrt(kl_dis))
    else:
        kl_loss = 1 - 1 / (tau + torch.log1p(kl_dis))
    return kl_loss


@MODELS.register_module()
class GDLoss_v1(nn.Module):
    """Gaussian based loss.

    Args:
        loss_type (str):  Type of loss.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    BAG_GD_LOSS = {'kld': kld_loss, 'bcd': bcd_loss, 'gwd': gwd_loss}

    def __init__(self,
                 loss_type,
                 fun='sqrt',
                 tau=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(GDLoss_v1, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'sqrt', '']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = xy_wh_r_2_xy_sigma
        self.fun = fun
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            # handle different dim of weight
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        if weight is not None:
            mask = (weight > 0).detach()
            pred = pred[mask]
            target = target[mask]

        pred = self.preprocess(pred)
        target = self.preprocess(target)

        return self.loss(
            pred, target, fun=self.fun, tau=self.tau, **
            _kwargs) * self.loss_weight
