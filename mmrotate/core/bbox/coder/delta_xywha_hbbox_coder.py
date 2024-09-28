# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import ROTATED_BBOX_CODERS
from ..transforms import norm_angle


@ROTATED_BBOX_CODERS.register_module()
class DeltaXYWHAHBBoxCoder(BaseBBoxCoder):
    """Delta XYWHA HBBox coder.

    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh, da)
    and decodes delta (dx, dy, dw, dh, da) back to original bbox
    (cx, cy, w, h, a).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by
            YOLOF. Default 32.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_range='oc',
                 norm_factor=None,
                 edge_swap=False,
                 clip_border=True,
                 add_ctr_clamp=False,
                 ctr_clamp=32):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.angle_range = angle_range
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5
        if self.angle_range in ['oc', 'le135', 'le90']:
            return bbox2delta(bboxes, gt_bboxes, self.means, self.stds,
                              self.angle_range, self.norm_factor,
                              self.edge_swap)
        else:
            raise NotImplementedError

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (torch.Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, num_classes * 5) or (B, N, 5) or
               (N, num_classes * 5) or (N, 5). Note N = num_anchors * W * H
               when rois is a grid of anchors.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 5), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        assert bboxes.size(-1) == 4
        assert pred_bboxes.size(-1) == 5
        if self.angle_range in ['oc', 'le135', 'le90']:
            return delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                              wh_ratio_clip, self.add_ctr_clamp,
                              self.ctr_clamp, self.angle_range,
                              self.norm_factor, self.edge_swap)
        else:
            raise NotImplementedError


@mmcv.jit(coderize=True)
def bbox2delta(proposals,
               gt,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               angle_range='oc',
               norm_factor=None,
               edge_swap=False):
    """We usually compute the deltas of x, y, w, h, a of proposals w.r.t ground
    truth bboxes to get regression target. This is the inverse function of
    :func:`delta2bbox`.

    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.

    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh, da.
    """
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx, gy, gw, gh, ga = gt.unbind(dim=-1)

    if edge_swap:
        dtheta1 = norm_angle(ga, angle_range)
        dtheta2 = norm_angle(ga + np.pi / 2, angle_range)
        abs_dtheta1 = torch.abs(dtheta1)
        abs_dtheta2 = torch.abs(dtheta2)
        gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
        gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
        ga = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
        dw = torch.log(gw_regular / pw)
        dh = torch.log(gh_regular / ph)
    else:
        ga = norm_angle(ga, angle_range)
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

    dx = (gx - px) / pw
    dy = (gy - py) / ph

    if norm_factor:
        da = ga / (norm_factor * np.pi)
    else:
        da = ga

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


@mmcv.jit(coderize=True)
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               wh_ratio_clip=16 / 1000,
               add_ctr_clamp=False,
               ctr_clamp=32,
               angle_range='oc',
               norm_factor=None,
               edge_swap=False):
    """Apply deltas to shift/scale base boxes. Typically the rois are anchor
    or proposed bounding boxes and the deltas are network outputs used to
    shift/scale those boxes. This is the inverse function of
    :func:`bbox2delta`.

    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 5) or (N, 5). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by
            YOLOF. Default 32.
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """
    means = deltas.new_tensor(means).view(1,
                                          -1).repeat(1,
                                                     deltas.size(-1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(-1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[..., 0::5]
    dy = denorm_deltas[..., 1::5]
    dw = denorm_deltas[..., 2::5]
    dh = denorm_deltas[..., 3::5]
    da = denorm_deltas[..., 4::5]

    if norm_factor:
        da *= norm_factor * np.pi

    x1, y1, x2, y2 = rois.unbind(dim=-1)
    # Compute center of each roi
    px = ((x1 + x2) * 0.5).unsqueeze(-1).expand_as(dx)
    py = ((y1 + y2) * 0.5).unsqueeze(-1).expand_as(dy)
    # Compute width/height of each roi
    pw = (x2 - x1).unsqueeze(-1).expand_as(dw)
    ph = (y2 - y1).unsqueeze(-1).expand_as(dh)

    dx_width = pw * dx
    dy_height = ph * dy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    ga = norm_angle(da, angle_range)
    if edge_swap:
        w_regular = torch.where(gw > gh, gw, gh)
        h_regular = torch.where(gw > gh, gh, gw)
        theta_regular = torch.where(gw > gh, ga, ga + np.pi / 2)
        theta_regular = norm_angle(theta_regular, angle_range)
        return torch.stack([gx, gy, w_regular, h_regular, theta_regular],
                           dim=-1).view_as(deltas)
    else:
        return torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
