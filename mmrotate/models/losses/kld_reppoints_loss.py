# Copyright (c) SJTU. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from mmrotate.core import GaussianMixture, gt2gaussian
from ..builder import ROTATED_LOSSES


def kld_single2single(g1, g2):
    """Compute Kullback-Leibler Divergence.

    Args:
        g1 (dict[str, torch.Tensor]): Gaussian distribution 1.
        g2 (torch.Tensor): Gaussian distribution 2.

    Returns:
        torch.Tensor: Kullback-Leibler Divergence.
    """
    p_mu = g1.mu
    p_var = g1.var
    assert p_mu.dim() == 3 and p_mu.size()[1] == 1
    assert p_var.dim() == 4 and p_var.size()[1] == 1
    p_mu = p_mu.squeeze(1)
    p_var = p_var.squeeze(1)
    t_mu, t_var = g2
    delta = (p_mu - t_mu).unsqueeze(-1)
    t_inv = torch.inverse(t_var)
    term1 = delta.transpose(-1, -2).matmul(t_inv).matmul(delta).squeeze(-1)
    term2 = torch.diagonal(
        t_inv.matmul(p_var),
        dim1=-2,
        dim2=-1).sum(dim=-1, keepdim=True) + \
        torch.log(torch.det(t_var) / torch.det(p_var)).reshape(-1, 1)

    return 0.5 * (term1 + term2) - 1


@weighted_loss
def kld_loss(pred, target, eps=1e-6):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Convexes with shape (N, 9, 2).
        target (torch.Tensor): Polygons with shape (N, 4, 2).
        eps (float): Defaults to 1e-6.

    Returns:
        torch.Tensor: Kullback-Leibler Divergence loss.
    """
    pred = pred.reshape(-1, 9, 2)
    target = target.reshape(-1, 4, 2)

    assert pred.size()[0] == target.size()[0] and target.numel() > 0
    gmm = GaussianMixture(n_components=1, requires_grad=True)
    gmm.fit(pred)
    kld = kld_single2single(gmm, gt2gaussian(target))
    kl_agg = kld.clamp(min=eps)
    loss = 1 - 1 / (2 + torch.sqrt(kl_agg))

    return loss


@ROTATED_LOSSES.register_module()
class KLDRepPointsLoss(nn.Module):
    """Kullback-Leibler Divergence loss for RepPoints.

    Args:
        eps (float): Defaults to 1e-6.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    """

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(KLDRepPointsLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

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

        Returns:
            loss (torch.Tensor)
        """
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * kld_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
