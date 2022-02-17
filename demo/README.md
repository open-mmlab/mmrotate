# Rotation Detection Demo

We provide a demo script to test a single image.

```shell
python demo/image_demo.py \
    ${IMG_ROOT} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${OUTPUT_ROOT}]
```

Examples:

```shell
    python demo/image_demo.py \
        demo/demo.jpg \
        work_dirs/oriented_rcnn_r50_fpn_1x_dota_v3/oriented_rcnn_r50_fpn_1x_dota_v3.py \
        work_dirs/oriented_rcnn_r50_fpn_1x_dota_v3/epoch_12.pth \
        demo/vis.jpg
```
