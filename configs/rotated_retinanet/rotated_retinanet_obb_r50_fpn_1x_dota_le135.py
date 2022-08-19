_base_ = 'rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'

angle_version = 'le135'

model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=angle_version),
        bbox_coder=dict(
            angle_version=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True)))
