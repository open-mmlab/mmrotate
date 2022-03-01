# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from math import ceil

import numpy as np


def get_multiscale_patch(sizes, steps, ratios):
    """Get multiscale patch sizes and steps.

    Args:
        sizes (list): A list of patch sizes.
        steps (list): A list of steps to slide patches.
        ratios (list): Multiscale ratios. devidie to each size and step and
            generate patches in new scales.

    Returns:
        new_sizes (list): A list of multiscale patch sizes.
        new_steps (list): A list of steps corresponding to new_sizes.
    """
    assert len(sizes) == len(steps), 'The length of `sizes` and `steps`' \
                                     'should be the same.'
    new_sizes, new_steps = [], []
    size_steps = list(zip(sizes, steps))
    for (size, step), ratio in product(size_steps, ratios):
        new_sizes.append(int(size / ratio))
        new_steps.append(int(step / ratio))
    return new_sizes, new_steps


def slide_window(width, height, sizes, steps, img_rate_thr=0.6):
    """Slide windows in images and get window position.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        sizes (list): List of window's sizes.
        steps (list): List of window's steps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        np.ndarray: Information of valid windows.
    """
    assert 1 >= img_rate_thr >= 0, 'The `in_rate_thr` should lie in 0~1'
    windows = []
    # Sliding windows.
    for size, step in zip(sizes, steps):
        assert size > step, 'Size should large than step'

        x_num = 1 if width <= size else ceil((width - size) / step + 1)
        x_start = [step * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size > width:
            x_start[-1] = width - size

        y_num = 1 if height <= size else ceil((height - size) / step + 1)
        y_start = [step * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size > height:
            y_start[-1] = height - size

        start = np.array(list(product(x_start, y_start)), dtype=np.int64)
        windows.append(np.concatenate([start, start + size], axis=1))
    windows = np.concatenate(windows, axis=0)

    # Calculate the rate of image part in each window.
    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates >= img_rate_thr).any():
        img_rates[img_rates == img_rates.max()] = 1
    return windows[img_rates >= img_rate_thr]
