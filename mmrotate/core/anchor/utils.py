# Copyright (c) OpenMMLab. All rights reserved.
def rotated_anchor_inside_flags(flat_anchors,
                                valid_flags,
                                img_shape,
                                allowed_border=0):
    """Check whether the rotated anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 5).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a valid
        range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        cx, cy = (flat_anchors[:, i] for i in range(2))
        inside_flags = \
            valid_flags & \
            (cx >= -allowed_border) & \
            (cy >= -allowed_border) & \
            (cx < img_w + allowed_border) & \
            (cy < img_h + allowed_border)
    else:
        inside_flags = valid_flags

    return inside_flags
