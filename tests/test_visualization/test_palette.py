# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.datasets import DOTADataset
from mmrotate.visualization import get_palette


def test_palette():

    # test list
    palette = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    palette_ = get_palette(palette, 3)
    for color, color_ in zip(palette, palette_):
        assert color == color_

    # test tuple
    palette = get_palette((1, 2, 3), 3)
    assert len(palette) == 3
    for color in palette:
        assert color == (1, 2, 3)

    # test color str
    palette = get_palette('red', 3)
    assert len(palette) == 3
    for color in palette:
        assert color == (255, 0, 0)

    # test dataset str
    palette = get_palette('dota', len(DOTADataset.METAINFO['classes']))
    assert len(palette) == len(DOTADataset.METAINFO['classes'])
    assert palette[0] == (165, 42, 42)

    # test random
    palette1 = get_palette('random', 3)
    palette2 = get_palette(None, 3)
    for color1, color2 in zip(palette1, palette2):
        assert isinstance(color1, tuple)
        assert isinstance(color2, tuple)
        assert color1 == color2
