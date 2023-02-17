# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import numpy as np
import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from mmdet.structures.bbox import HorizontalBoxes, get_box_tensor
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes
from ....structures.bbox.transforms import norm_angle


@TASK_UTILS.register_module()
class DeltaXYWHTHBBoxCoder(BaseBBoxCoder):
    """Delta XYWHT HBBox coder. This coder encodes bbox (x1, y1, x2, y2) into
    delta (dx, dy, dw, dh, dt) and decodes delta (dx, dy, dw, dh, dt) back to
    original bbox (cx, cy, w, h, t).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates.
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        angle_version (str): Angle representations. Defaults to 'oc'.
        norm_factor (float, optional): Regularization factor of angle.
        edge_swap (bool): Whether swap the edge if w < h.
            Defaults to False.
        clip_border (bool): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF.
            Defaults to False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by
            YOLOF. Defaults to 32.
        use_box_type (bool): Whether to warp decoded boxes with the
            box type data structure. Defaults to True.
    """
    encode_size = 5

    def __init__(self,
                 target_means: Sequence[float] = (0., 0., 0., 0., 0.),
                 target_stds: Sequence[float] = (1., 1., 1., 1., 1.),
                 angle_version: str = 'oc',
                 norm_factor: Optional[float] = None,
                 edge_swap: bool = False,
                 clip_border: bool = True,
                 add_ctr_clamp: bool = False,
                 ctr_clamp: int = 32,
                 use_box_type=True) -> None:
        super().__init__(use_box_type=use_box_type)
        self.means = target_means
        self.stds = target_stds
        self.angle_version = angle_version
        assert self.angle_version in ['oc', 'le135', 'le90', 'r360']
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes: HorizontalBoxes,
               gt_bboxes: RotatedBoxes) -> Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (:obj:`HorizontalBoxes` or Tensor): Source boxes,
                e.g.,object proposals.
            gt_bboxes (:obj:`RotatedBoxes`): Target of the transformation,
                e.g., ground-truth boxes.

        Returns:
            Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5
        return bbox2delta(bboxes, gt_bboxes, self.means, self.stds,
                          self.angle_version, self.norm_factor, self.edge_swap)

    def decode(
            self,
            bboxes: Union[HorizontalBoxes, Tensor],
            pred_bboxes: Tensor,
            max_shape: Optional[Sequence[int]] = None,
            wh_ratio_clip: float = 16 / 1000) -> Union[RotatedBoxes, Tensor]:
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (:obj:`HorizontalBoxes` or Tensor): Basic boxes.
                Shape (B, N, 4) or (N, 4). In two stage detectors and refine
                single stage detectors, the bboxes can be Tensor.
            pred_bboxes (Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, num_classes * 5) or (B, N, 5) or
                (N, num_classes * 5) or (N, 5). Note N = num_anchors * W * H
                when rois is a grid of anchors.
            max_shape (Sequence[int] or Tensor or Sequence[
                Sequence[int]], optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If bboxes shape is (B, N, 5), then
                the max_shape should be a Sequence[Sequence[int]]
                and the length of max_shape should also be B.
            wh_ratio_clip (float): The allowed ratio between
                width and height.

        Returns:
            Union[:obj:`RotatedBoxes`, Tensor]: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        assert bboxes.size(-1) == 4
        assert pred_bboxes.size(-1) == 5
        bboxes = get_box_tensor(bboxes)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    wh_ratio_clip, self.add_ctr_clamp,
                                    self.ctr_clamp, self.angle_version,
                                    self.norm_factor, self.edge_swap)
        if self.use_box_type:
            assert decoded_bboxes.size(-1) == 5, \
                ('Cannot warp decoded boxes with box type when decoded'
                 'boxes have shape of (N, num_classes * 5)')
            decoded_bboxes = RotatedBoxes(decoded_bboxes)
        return decoded_bboxes


def bbox2delta(proposals: HorizontalBoxes,
               gts: RotatedBoxes,
               means: Sequence[float] = (0., 0., 0., 0., 0.),
               stds: Sequence[float] = (1., 1., 1., 1., 1.),
               angle_version: str = 'oc',
               norm_factor: Optional[float] = None,
               edge_swap: bool = False) -> Tensor:
    """We usually compute the deltas of x, y, w, h, t of proposals w.r.t ground
    truth bboxes to get regression target.

    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (:obj:`HorizontalBoxes`): Boxes to be transformed,
            shape (N, ..., 4)
        gts (:obj:`RotatedBoxes`): Gt bboxes to be used as base,
            shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
        angle_version (str): Angle definition. Defaults to 'oc'.
        norm_factor (float, optional): Regularization factor of angle.
        edge_swap (bool): Whether swap the edge if w < h.
            Defaults to False.
    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
        dw, dh, dt.
    """
    proposals = proposals.tensor
    proposals = proposals.float()
    gts = gts.regularize_boxes(angle_version)
    gts = gts.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx, gy, gw, gh, gt = gts.unbind(dim=-1)

    if edge_swap:
        dtheta1 = norm_angle(gt, angle_version)
        dtheta2 = norm_angle(gt + np.pi / 2, angle_version)
        abs_dtheta1 = torch.abs(dtheta1)
        abs_dtheta2 = torch.abs(dtheta2)
        gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
        gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
        gt = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
        dw = torch.log(gw_regular / pw)
        dh = torch.log(gh_regular / ph)
    else:
        gt = norm_angle(gt, angle_version)
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

    dx = (gx - px) / pw
    dy = (gy - py) / ph

    if norm_factor:
        dt = gt / (norm_factor * np.pi)
    else:
        dt = gt

    deltas = torch.stack([dx, dy, dw, dh, dt], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois: Tensor,
               deltas: Tensor,
               means: Sequence[float] = (0., 0., 0., 0., 0.),
               stds: Sequence[float] = (1., 1., 1., 1., 1.),
               wh_ratio_clip: float = 16 / 1000,
               add_ctr_clamp: bool = False,
               ctr_clamp: int = 32,
               angle_version: str = 'oc',
               norm_factor: Optional[float] = None,
               edge_swap: bool = False) -> Tensor:
    """Apply deltas to shift/scale base boxes. Typically the rois are anchor
    or proposed bounding boxes and the deltas are network outputs used to
    shift/scale those boxes. This is the inverse function of
    :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed.
            Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 5) or (N, 5). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Defaults to (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Defaults to (1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF.
            Defaults to False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by
            YOLOF. Defaults to 32.
        angle_version (str): Angle representations. Defaults to 'oc'.
        norm_factor (float, optional): Regularization factor of angle.
        edge_swap (bool): Whether swap the edge if w < h.
            Defaults to False.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5),
        where 5 represent cx, cy, w, h, t.
    """
    num_bboxes = deltas.size(0)
    if num_bboxes == 0:
        return deltas

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 5))
    denorm_deltas = reshaped_deltas * stds + means

    dx = denorm_deltas[..., 0::5]
    dy = denorm_deltas[..., 1::5]
    dw = denorm_deltas[..., 2::5]
    dh = denorm_deltas[..., 3::5]
    dt = denorm_deltas[..., 4::5]

    if norm_factor:
        dt *= norm_factor * np.pi

    # Compute center of each roi
    px = ((rois[..., None, None, 0] + rois[..., None, None, 2]) * 0.5)
    py = ((rois[..., None, None, 1] + rois[..., None, None, 3]) * 0.5)
    # Compute width/height of each roi
    pw = (rois[..., None, None, 2] - rois[..., None, None, 0])
    ph = (rois[..., None, None, 3] - rois[..., None, None, 1])

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
    gt = norm_angle(dt, angle_version)
    if edge_swap:
        w_regular = torch.where(gw > gh, gw, gh)
        h_regular = torch.where(gw > gh, gh, gw)
        theta_regular = torch.where(gw > gh, gt, gt + np.pi / 2)
        theta_regular = norm_angle(theta_regular, angle_version)
        decoded_bbox = torch.stack(
            [gx, gy, w_regular, h_regular, theta_regular],
            dim=-1).view_as(deltas)
    else:
        decoded_bbox = torch.stack([gx, gy, gw, gh, gt],
                                   dim=-1).view_as(deltas)

    return decoded_bbox
