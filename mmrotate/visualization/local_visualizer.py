# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import cv2
import numpy as np
import torch
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from mmdet.visualization import DetLocalVisualizer, jitter_color
from mmdet.visualization.palette import _get_adaptive_scales
from mmengine.structures import InstanceData

from mmrotate.registry import VISUALIZERS
from .palette import get_palette


@VISUALIZERS.register_module()
class RotLocalVisualizer(DetLocalVisualizer):
    """MMRotate Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.
    """

    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.
        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]

            # convert to qbox
            polygons = bboxes.convert_to('qbox').tensor
            polygons = polygons.reshape(-1, 4, 2).numpy()
            polygons = [p for p in polygons]
            self.draw_polygons(
                polygons,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)

            positions = bboxes.centers + self.line_width
            scales = _get_adaptive_scales(bboxes.areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(13 * scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(np.bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

            if len(labels) > 0 and \
                    ('bboxes' not in instances or
                     instances.bboxes.sum() == 0):
                # instances.bboxes.sum()==0 represent dummy bboxes.
                # A typical example of SOLO does not exist bbox branch.
                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, (pos, label) in enumerate(zip(positions, labels)):
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                    if 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        label_text += f': {score}'

                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(13 * scales[i]),
                        horizontal_alignments='center',
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])
        return self.get_image()
