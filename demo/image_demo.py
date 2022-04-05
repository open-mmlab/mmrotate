# Copyright (c) OpenMMLab. All rights reserved.
"""Inference on single image.

Example:


```
wget -P checkpoint https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth  # noqa: E501, E261.
python demo/image_demo.py \
    demo/demo.jpg \
    configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py \
    work_dirs/oriented_rcnn_r50_fpn_1x_dota_v3/epoch_12.pth
```
"""  # nowq

from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
