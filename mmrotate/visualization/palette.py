# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import mmcv
import numpy as np
from mmengine.utils import is_str


def get_palette(palette: Union[List[tuple], str, tuple],
                num_classes: int) -> List[Tuple[int]]:
    """Get palette from various inputs.

    Args:
        palette (list[tuple] | str | tuple): palette inputs.
        num_classes (int): the number of classes.
    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)

    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == 'random' or palette is None:
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif palette == 'dota':
        from mmrotate.datasets import DOTADataset
        dataset_palette = DOTADataset.METAINFO['palette']
    elif palette == 'sar':
        from mmrotate.datasets import SARDataset
        dataset_palette = SARDataset.METAINFO['palette']
    elif palette == 'hrsc':
        from mmrotate.datasets import HRSCDataset
        dataset_palette = HRSCDataset.METAINFO['palette']
    elif is_str(palette):
        dataset_palette = [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')

    assert len(dataset_palette) >= num_classes, \
        'The length of palette should not be less than `num_classes`.'
    return dataset_palette
