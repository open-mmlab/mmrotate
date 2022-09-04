# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder

from mmrotate.core.bbox.structures import RotatedBoxes
from mmrotate.registry import TASK_UTILS
from ..transforms import norm_angle


@TASK_UTILS.register_module()
class DeltaXYWHTRBBoxCoder(BaseBBoxCoder):
    """Delta XYWHT RBBox coder. This coder is used for rotated objects
    detection (for example on task1 of DOTA dataset). this coder encodes bbox
    (xc, yc, w, h, t) into delta (dx, dy, dw, dh, da) and decodes delta (dx,
    dy, dw, dh, da) back to original bbox (xc, yc, w, h, t).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        angle_version (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.
        proj_xy (bool, optional): Whether project x and y according to angle.
            Defaults to False.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by
            YOLOF. Default 32.
    """
    encode_size = 5

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_version='oc',
                 norm_factor=None,
                 edge_swap=False,
                 proj_xy=False,
                 add_ctr_clamp=False,
                 ctr_clamp=32):
        super().__init__()
        self.means = target_means
        self.stds = target_stds
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.angle_version = angle_version
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.proj_xy = proj_xy

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (:obj:`RotatedBoxes`): Source boxes, e.g., object proposals.
            gt_bboxes (:obj:`RotatedBoxes`): Target of the transformation,
                e.g., ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 5
        assert gt_bboxes.size(-1) == 5
        if self.angle_version in ['oc', 'le135', 'le90']:
            return bbox2delta(bboxes, gt_bboxes, self.means, self.stds,
                              self.angle_version, self.norm_factor,
                              self.edge_swap, self.proj_xy)
        else:
            raise NotImplementedError

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (:obj:`RotatedBoxes`): Basic boxes. Shape (B, N, 5) or \
                (N, 5)
            pred_bboxes (torch.Tensor): Encoded offsets with respect to each \
                roi. Has shape (B, N, num_classes * 5) or (B, N, 5) or \
               (N, num_classes * 5) or (N, 5). Note N = num_anchors * W * H \
               when rois is a grid of anchors.
            max_shape (Sequence[int] or torch.Tensor or Sequence[ \
               Sequence[int]],optional): Maximum bounds for boxes, specifies \
               (H, W, C) or (H, W). If bboxes shape is (B, N, 5), then \
               the max_shape should be a Sequence[Sequence[int]] \
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            :obj:`RotatedBoxes`: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        if self.angle_version in ['oc', 'le135', 'le90']:
            return delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                              max_shape, wh_ratio_clip, self.add_ctr_clamp,
                              self.ctr_clamp, self.angle_version,
                              self.norm_factor, self.edge_swap, self.proj_xy)
        else:
            raise NotImplementedError


def bbox2delta(proposals,
               gt,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               angle_version='oc',
               norm_factor=None,
               edge_swap=False,
               proj_xy=False):
    """We usually compute the deltas of x, y, w, h, a of proposals w.r.t ground
    truth bboxes to get regression target. This is the inverse function of
    :func:`delta2bbox`.

    Args:
        proposals (:obj:`RotatedBoxes`): Boxes to be transformed,
            shape (N, ..., 5)
        gt (:obj:`RotatedBoxes`): Gt bboxes to be used as base,
            shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
        angle_version (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.
        proj_xy (bool, optional): Whether project x and y according to angle.
            Defaults to False.

    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh, da.
    """
    assert proposals.size() == gt.size()
    proposals = proposals.tensor
    gt = gt.regularize_boxes(angle_version)
    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph, pa = proposals.unbind(dim=-1)
    gx, gy, gw, gh, ga = gt.unbind(dim=-1)

    if proj_xy:
        dx = (torch.cos(pa) * (gx - px) + torch.sin(pa) * (gy - py)) / pw
        dy = (-torch.sin(pa) * (gx - px) + torch.cos(pa) * (gy - py)) / ph
    else:
        dx = (gx - px) / pw
        dy = (gy - py) / ph

    if edge_swap:
        dtheta1 = norm_angle(ga - pa, angle_version)
        dtheta2 = norm_angle(ga - pa + np.pi / 2, angle_version)
        abs_dtheta1 = torch.abs(dtheta1)
        abs_dtheta2 = torch.abs(dtheta2)
        gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
        gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
        da = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
        dw = torch.log(gw_regular / pw)
        dh = torch.log(gh_regular / ph)
    else:
        da = norm_angle(ga - pa, angle_version)
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

    if norm_factor:
        da /= norm_factor * np.pi

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               add_ctr_clamp=False,
               ctr_clamp=32,
               angle_version='oc',
               norm_factor=None,
               edge_swap=False,
               proj_xy=False):
    """Apply deltas to shift/scale base boxes. Typically the rois are anchor
    or proposed bounding boxes and the deltas are network outputs used to
    shift/scale those boxes. This is the inverse function of
    :func:`bbox2delta`.

    Args:
        rois (:obj:`RotatedBoxes`): Boxes to be transformed. Has shape (N, 5).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 5) or (N, 5). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1.).
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
           (H, W, C) or (H, W). If bboxes shape is (B, N, 5), then
           the max_shape should be a Sequence[Sequence[int]]
           and the length of max_shape should also be B.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by
            YOLOF. Default 32.
        angle_version (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.
        proj_xy (bool, optional): Whether project x and y according to angle.
            Defaults to False.

    Returns:
        :obj:`RotatedBoxes`: Boxes with shape (N, num_classes * 5) or (N, 5),
            where 5 represent cx, cy, w, h, a.
    """
    num_bboxes = deltas.size(0)
    if num_bboxes == 0:
        return RotatedBoxes(deltas)

    rois = rois.tensor
    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 5))
    denorm_deltas = reshaped_deltas * stds + means

    dx = denorm_deltas[..., 0]
    dy = denorm_deltas[..., 1]
    dw = denorm_deltas[..., 2]
    dh = denorm_deltas[..., 3]
    da = denorm_deltas[..., 4]
    if norm_factor:
        da *= norm_factor * np.pi
    # Compute center of each roi

    px = rois[..., None, 0]
    py = rois[..., None, 1]
    # Compute width/height of each roi
    pw = rois[..., None, 2]
    ph = rois[..., None, 3]
    # Compute rotated angle of each roi
    pa = rois[..., None, 4]
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
    if proj_xy:
        gx = dx * pw * torch.cos(pa) - dy * ph * torch.sin(pa) + px
        gy = dx * pw * torch.sin(pa) + dy * ph * torch.cos(pa) + py
    else:
        gx = px + dx_width
        gy = py + dy_height
    # Compute angle
    ga = norm_angle(pa + da, angle_version)
    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)

    if edge_swap:
        w_regular = torch.where(gw > gh, gw, gh)
        h_regular = torch.where(gw > gh, gh, gw)
        theta_regular = torch.where(gw > gh, ga, ga + np.pi / 2)
        theta_regular = norm_angle(theta_regular, angle_version)
        decoded_bbox = torch.stack(
            [gx, gy, w_regular, h_regular, theta_regular],
            dim=-1).view_as(deltas)
    else:
        decoded_bbox = torch.stack([gx, gy, gw, gh, ga],
                                   dim=-1).view(deltas.size())
    return RotatedBoxes(decoded_bbox)
