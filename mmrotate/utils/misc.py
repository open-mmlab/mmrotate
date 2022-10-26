# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import warnings

from mmengine.config import Config


def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.

    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def get_test_pipeline_cfg(cfg):
    """Get the test dataset pipeline from entire config.

    Args:
        cfg (str or :obj:`ConfigDict`): the entire config. Can be a config
            file or a ``ConfigDict``.

    Returns:
        :obj:`ConfigDict`: the config of test dataset.
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    dataset_cfg = cfg.test_dataloader.dataset
    test_pipeline = dataset_cfg.get('pipeline', None)
    # handle dataset wrapper
    if test_pipeline is None:
        if 'dataset' in dataset_cfg:
            test_pipeline = dataset_cfg.dataset.pipeline
        # handle dataset wrappers like ConcatDataset
        elif 'datasets' in dataset_cfg:
            test_pipeline = dataset_cfg.datasets[0].pipeline
        else:
            raise RuntimeError('Cannot find `pipeline` in `test_dataloader`')
    return test_pipeline
