# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash

import mmrotate


def collect_env():
    """Collect environment information."""
    env_info = collect_basic_env()
    env_info['MMRotate'] = (
        mmrotate.__version__ + '+' + get_git_hash(digits=7))
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
