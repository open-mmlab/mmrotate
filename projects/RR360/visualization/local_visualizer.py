# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import numpy as np
import torch
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from mmdet.visualization import DetLocalVisualizer, jitter_color
from mmdet.visualization.palette import _get_adaptive_scales
from mmengine.structures import InstanceData
from projects.RR360.structures.bbox import RotatedBoxes
from torch import Tensor

from mmrotate.registry import VISUALIZERS
# from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes
from mmrotate.structures.bbox import QuadriBoxes
from mmrotate.visualization.palette import get_palette


@VISUALIZERS.register_module()
class RR360LocalVisualizer(DetLocalVisualizer):
    """RR360LocalVisualizer Local Visualizer, show the A B C D Points.

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

            if isinstance(bboxes, Tensor):
                if bboxes.size(-1) == 5:
                    bboxes = RotatedBoxes(bboxes)
                elif bboxes.size(-1) == 8:
                    bboxes = QuadriBoxes(bboxes)
                else:
                    raise TypeError(
                        'Require the shape of `bboxes` to be (n, 5) '
                        'or (n, 8), but get `bboxes` with shape being '
                        f'{bboxes.shape}.')

            bboxes = bboxes.cpu()
            polygons = bboxes.convert_to('qbox').tensor
            polygons = polygons.reshape(-1, 4, 2)
            polygons = [p for p in polygons]
            # colors=[(0,0,0),(255,0,0),(0,255,0),(0,0,255)]
            self.draw_polygons(
                polygons,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)

            for i, (poly, label) in enumerate(zip(polygons, labels)):
                self.draw_points(
                    positions=bboxes.convert_to('qbox').tensor.reshape(
                        -1, 4, 2)[i],
                    colors=[(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
                    sizes=20,
                )
                self.draw_texts(
                    ['A', 'B', 'C', 'D'],
                    positions=bboxes.convert_to('qbox').tensor.reshape(
                        -1, 4, 2)[i],
                    colors=[(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
                    font_sizes=20,
                )

            positions = bboxes.centers + self.line_width
            scales = _get_adaptive_scales(bboxes.areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

                angle_text = 'angle ' + str(
                    np.round(180 * bboxes.numpy()[i][4] / np.pi, 1))

                label_text += ' ' + angle_text

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

            masks = masks.astype(bool)

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
        return self.get_image()
