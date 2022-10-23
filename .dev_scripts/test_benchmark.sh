CHECKPOINT_DIR=$1

echo 'configs/cfa/cfa-qbox_r50_fpn_1x_dota.py' &
python tools/test.py configs/cfa/cfa-qbox_r50_fpn_1x_dota.py $CHECKPOINT_DIR/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --cfg-option env_cfg.dist_cfg.port=29666  &
