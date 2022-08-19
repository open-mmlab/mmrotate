_base_ = 'rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'

angle_version = 'oc'

model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=angle_version),
<<<<<<< HEAD
<<<<<<< HEAD
        bbox_coder=dict(
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False)))
=======
        bbox_coder=dict(angle_version=angle_version)))
>>>>>>> 61dcdf7 (init)
=======
        bbox_coder=dict(
<<<<<<< HEAD
            angle_version=angle_version, edge_swap=False, proj_xy=False)))
>>>>>>> e8486b9 (fix bug)
=======
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False)))
>>>>>>> 0289589 (update configs & RBboxOverlaps2D & FakeRBboxOverlaps2D)
