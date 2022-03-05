#!/usr/bin/env bash

#python tools/train.py configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py
#python tools/train.py configs/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135.py
#python tools/train.py configs/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90.py
#python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90.py
#python tools/train.py configs/redet/redet_re50_refpn_fp16_1x_dota_le90.py

python -m torch.distributed.launch --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py work_dirs/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/epoch_12.pth --launcher pytorch
#python ./tools/test.py  \
#  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py \
#  work_dirs/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/epoch_12.pth --format-only \
#  --eval-options submission_dir=work_dirs/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/Task1_results
#
#python ./tools/test.py  \
#  configs/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90.py \
#  work_dirs/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/epoch_12.pth --format-only \
#  --eval-options submission_dir=work_dirs/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/Task1_results
#
#python ./tools/test.py  \
#  configs/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135.py \
#  work_dirs/s2anet_r50_fpn_fp16_1x_dota_le135/epoch_12.pth --format-only \
#  --eval-options submission_dir=work_dirs/s2anet_r50_fpn_fp16_1x_dota_le135/Task1_results
#
#python ./tools/test.py  \
#  configs/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90.py \
#  work_dirs/roi_trans_r50_fpn_fp16_1x_dota_le90/epoch_12.pth --format-only \
#  --eval-options submission_dir=work_dirs/roi_trans_r50_fpn_fp16_1x_dota_le90/Task1_results
#
#
#python ./tools/test.py  \
#  configs/redet/redet_re50_refpn_fp16_1x_dota_le90.py \
#  work_dirs/redet_re50_refpn_fp16_1x_dota_le90/epoch_12.pth --format-only \
#  --eval-options submission_dir=work_dirs/redet_re50_refpn_fp16_1x_dota_le90/Task1_results
#
#
