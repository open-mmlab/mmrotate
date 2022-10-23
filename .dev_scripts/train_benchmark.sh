echo 'configs/cfa/cfa-qbox_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/cfa/cfa-qbox_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/cfa/cfa-qbox_r50_fpn_40e_dota.py' &
python ./tools/train.py configs/cfa/cfa-qbox_r50_fpn_40e_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/convnext/rotated-retinanet-rbox-le90_convnext-tiny_fpn_kld-stable_adamw-1x_dota.py' &
python ./tools/train.py configs/convnext/rotated-retinanet-rbox-le90_convnext-tiny_fpn_kld-stable_adamw-1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/csl/rotated-retinanet-rbox-le90_r50_fpn_csl-gaussian_amp-1x_dota.py' &
python ./tools/train.py configs/csl/rotated-retinanet-rbox-le90_r50_fpn_csl-gaussian_amp-1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
# echo 'configs/gliding_vertex/gliding-vertex-rbox_r50_fpn_1x_dota.py' &
# python ./tools/train.py configs/gliding_vertex/gliding-vertex-rbox_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/gwd/rotated-retinanet-hbox-oc_r50_fpn_gwd_1x_dota.py' &
python ./tools/train.py configs/gwd/rotated-retinanet-hbox-oc_r50_fpn_gwd_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kfiou/rotated-retinanet-hbox-le90_r50_fpn_kfiou_1x_dota.py' &
python ./tools/train.py configs/kfiou/rotated-retinanet-hbox-le90_r50_fpn_kfiou_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kfiou/rotated-retinanet-hbox-oc_r50_fpn_kfiou_1x_dota.py' &
python ./tools/train.py configs/kfiou/rotated-retinanet-hbox-oc_r50_fpn_kfiou_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kfiou/rotated-retinanet-hbox-le135_r50_fpn_kfiou_1x_dota.py' &
python ./tools/train.py configs/kfiou/rotated-retinanet-hbox-le135_r50_fpn_kfiou_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kfiou/r3det-oc_r50_fpn_kfiou-ln_1x_dota.py' &
python ./tools/train.py configs/kfiou/r3det-oc_r50_fpn_kfiou-ln_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kld/rotated-retinanet-hbox-oc_r50_fpn_kld_1x_dota.py' &
python ./tools/train.py configs/kld/rotated-retinanet-hbox-oc_r50_fpn_kld_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kld/rotated-retinanet-hbox-oc_r50_fpn_kld-stable_1x_dota.py' &
python ./tools/train.py configs/kld/rotated-retinanet-hbox-oc_r50_fpn_kld-stable_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kld/rotated-retinanet-rbox-le90_r50_fpn_kld-stable_1x_dota.py' &
python ./tools/train.py configs/kld/rotated-retinanet-rbox-le90_r50_fpn_kld-stable_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kld/r3det-oc_r50_fpn_kld_1x_dota.py' &
python ./tools/train.py configs/kld/r3det-oc_r50_fpn_kld_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kld/r3det-oc_r50_fpn_kld-stable_1x_dota.py' &
python ./tools/train.py configs/kld/r3det-oc_r50_fpn_kld-stable_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kld/r3det-tiny-oc_r50_fpn_kld_1x_dota.py' &
python ./tools/train.py configs/kld/r3det-tiny-oc_r50_fpn_kld_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/kld/rotated-retinanet-rbox-le90_r50_fpn_kld-stable_adamw-1x_dota.py' &
python ./tools/train.py configs/kld/rotated-retinanet-rbox-le90_r50_fpn_kld-stable_adamw-1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_amp-1x_dota.py' &
python ./tools/train.py configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_amp-1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/r3det/r3det-oc_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/r3det/r3det-oc_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/r3det/r3det-tiny-oc_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/r3det/r3det-tiny-oc_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/redet/redet-le90_re50_refpn_amp-1x_dota.py' &
./tools/dist_train.sh configs/redet/redet-le90_re50_refpn_amp-1x_dota.py 1 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/redet/redet-le90_re50_refpn_1x_dota.py' &
./tools/dist_train.sh configs/redet/redet-le90_re50_refpn_1x_dota.py 1 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/roi_trans/roi-trans-le90_r50_fpn_amp-1x_dota.py' &
python ./tools/train.py configs/roi_trans/roi-trans-le90_r50_fpn_amp-1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/roi_trans/roi-trans-le90_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/roi_trans/roi-trans-le90_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/roi_trans/roi-trans-le90_swin-tiny_fpn_1x_dota.py' &
python ./tools/train.py configs/roi_trans/roi-trans-le90_swin-tiny_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_atss/rotated-atss-le90_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/rotated_atss/rotated-atss-le90_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_fcos/rotated-fcos-le90_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/rotated_fcos/rotated-fcos-le90_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_csl-gaussian_1x_dota.py' &
python ./tools/train.py configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_csl-gaussian_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_fcos/rotated-fcos-le90_r50_fpn_kld_1x_dota.py' &
python ./tools/train.py configs/rotated_fcos/rotated-fcos-le90_r50_fpn_kld_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_reppoints/rotated-reppoints-qbox_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/rotated_reppoints/rotated-reppoints-qbox_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_retinanet/rotated-retinanet-hbox-oc_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/rotated_retinanet/rotated-retinanet-hbox-oc_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota.py' &
python ./tools/train.py configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/rotated_retinanet/rotated-retinanet-rbox-le135_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/rotated_retinanet/rotated-retinanet-rbox-le135_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/s2anet/s2anet-le135_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/s2anet/s2anet-le135_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/s2anet/s2anet-le135_r50_fpn_amp-1x_dota.py' &
python ./tools/train.py configs/s2anet/s2anet-le135_r50_fpn_amp-1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/sasm_reppoints/sasm-reppoints-qbox_r50_fpn_1x_dota.py' &
python ./tools/train.py configs/sasm_reppoints/sasm-reppoints-qbox_r50_fpn_1x_dota.py --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
