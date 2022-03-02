#!/usr/bin/env bash
python tools/train.py configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py
python tools/train.py configs/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135.py
python tools/train.py configs/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90.py
python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90.py
python tools/train.py configs/redet/redet_re50_refpn_fp16_1x_dota_le90.py