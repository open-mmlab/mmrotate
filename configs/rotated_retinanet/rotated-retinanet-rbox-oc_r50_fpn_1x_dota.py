_base_ = 'rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py'

angle_version = 'oc'

model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=angle_version),
        bbox_coder=dict(
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False)))
