# Copyright (c) OpenMMLab. All rights reserved.
import os
os.system('python -m mim install "mmcv>=2.0.0rc4"')
os.system('python -m mim install mmengine')
os.system('python -m mim install "mmdet>=3.0.0"')
os.system('pip install git+https://github.com/cocodataset/panopticapi.git')
os.system('python -m mim install -e .')

from argparse import ArgumentParser
import gradio as gr
from mmdet.apis import DetInferencer, inference_detector

# from mmyolo.utils import switch_to_deploy
# from mmyolo.utils.misc import get_file_list


mmrotate_det_dict = {
    'RTMDet-T': 'rotated_rtmdet_tiny-3x-dota_ms',
    'RTMDet-L': 'rotated_rtmdet_l-coco_pretrain-3x-dota_ms',
}

DEFAULT_MMRotate_Det = 'RTMDet-T'

merged_dict = {}
merged_dict.update(mmrotate_det_dict)



def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def inference(input, model_str):
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()

    scope = 'mmrotate'

    det_inferencer = DetInferencer(merged_dict[model_str], scope=scope)
    # switch_to_deploy(det_inferencer.model)

    result = inference_detector(det_inferencer.model, input)

    det_inferencer.visualizer.add_datasample(
        'image',
        input,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        pred_score_thr=args.score_thr)

    return det_inferencer.visualizer.get_image()


DESCRIPTION = '''# MMRotate

#### This is an official demo for MMRotate. \n

Note: The first time running requires downloading the weights, please wait a moment.
'''

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab('MMRotate Demo'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image', type='numpy')
                with gr.Row():
                    model_name = gr.Dropdown(
                        list(list(mmrotate_det_dict.keys())),
                        value=DEFAULT_MMRotate_Det,
                        label='Model')
                with gr.Row():
                    run_button = gr.Button(value='Run')
            with gr.Column():
                visualization = gr.Image(label='Result', type='numpy')

        with gr.Row():
            # paths, _ = get_file_list('demo')
            example_images = gr.Dataset(
                components=[input_image], samples=[
                os.path.join("examples", e)
                for e in os.listdir("examples")
            ],)

        run_button.click(
            fn=inference,
            inputs=[input_image, model_name],
            outputs=[
                visualization,
            ])
        example_images.click(
            fn=set_example_image, inputs=example_images, outputs=input_image)

demo.queue().launch(show_api=False)