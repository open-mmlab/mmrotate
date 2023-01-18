# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import mmcv
import torch
from mmdet.apis import inference_detector, init_detector
from ts.torch_handler.base_handler import BaseHandler

import mmrotate  # noqa: F401


class MMRotateHandler(BaseHandler):
    """MMRotate handler to load torchscript or eager mode [state_dict]
    models."""
    threshold = 0.5

    def initialize(self, context):
        """Load the model.pt file and initialize the MMRotate model object.

        Args:
            context (context): JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_detector(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        """Convert the request input to a ndarray.

        Args :
            data (list): List of the data from the request input.

        Returns:
            list[ndarray]: The list of ndarray data of the input
        """
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        """Predict the results given input request.

        Args:
            data (list[ndarray]): The list of a ndarray which are ready to
                process.

        Returns:
            list[Tensor] : The list of results from the inference.
        """
        results = inference_detector(self.model, data)
        return results

    def postprocess(self, data):
        """Convert the output from the inference and converts into a Torchserve
        supported response output.

        Args:
            data (list[Tensor]): The list of results received from the
                predicted output of the model.

        Returns:
            list[dict]: The list of the predicted output that can be converted
                to json format.
        """
        output = []
        for image_index, image_result in enumerate(data):
            output.append([])
            if isinstance(image_result, tuple):
                bbox_result, segm_result = image_result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = image_result, None

            for class_index, class_result in enumerate(bbox_result):
                class_name = self.model.classes[class_index]
                for bbox in class_result:
                    bbox_coords = bbox[:-1].tolist()
                    score = float(bbox[-1])
                    if score >= self.threshold:
                        output[image_index].append({
                            'class_name': class_name,
                            'bbox': bbox_coords,
                            'score': score
                        })

        return output
