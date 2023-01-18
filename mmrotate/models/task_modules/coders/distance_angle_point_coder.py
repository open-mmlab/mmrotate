# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox.transforms import norm_angle


@TASK_UTILS.register_module()
class DistanceAnglePointCoder(BaseBBoxCoder):
    """Distance Angle Point BBox coder.

    This coder encodes gt bboxes (x, y, w, h, theta) into (top, bottom, left,
    right, theta) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border=True, angle_version='oc'):
        super(BaseBBoxCoder, self).__init__()
        self.clip_border = clip_border
        self.angle_version = angle_version

    def encode(self, points, gt_bboxes, max_dis=None, eps=0.1):
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 5), The format is "xywha"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 5).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 5
        return self.obb2distance(points, gt_bboxes, max_dis, eps)

    def decode(self, points, pred_bboxes, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries and angle (left, top, right, bottom, angle).
                Shape (B, N, 5) or (N, 5)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 5) or (B, N, 5)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 5
        if self.clip_border is False:
            max_shape = None
        return self.distance2obb(points, pred_bboxes, max_shape,
                                 self.angle_version)

    def obb2distance(self, points, distance, max_dis=None, eps=None):
        ctr, wh, angle = torch.split(distance, [2, 2, 1], dim=1)

        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=1).reshape(-1, 2, 2)

        offset = points - ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = wh[..., 0], wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        if max_dis is not None:
            left = left.clamp(min=0, max=max_dis - eps)
            top = top.clamp(min=0, max=max_dis - eps)
            right = right.clamp(min=0, max=max_dis - eps)
            bottom = bottom.clamp(min=0, max=max_dis - eps)
        return torch.stack((left, top, right, bottom, angle.squeeze(-1)), -1)

    def distance2obb(self,
                     points,
                     distance,
                     max_shape=None,
                     angle_version='oc'):
        distance, angle = distance.split([4, 1], dim=-1)

        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)

        rot_matrix = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle],
                               dim=-1)
        rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)

        wh = distance[..., :2] + distance[..., 2:]
        offset_t = (distance[..., 2:] - distance[..., :2]) / 2
        offset = torch.matmul(rot_matrix, offset_t[..., None]).squeeze(-1)
        ctr = points[..., :2] + offset

        angle_regular = norm_angle(angle, angle_version)
        return torch.cat([ctr, wh, angle_regular], dim=-1)
