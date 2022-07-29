# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List, Optional

import mmcv
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import FileClient

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DOTADataset(BaseDataset):
    """DOTA dataset for detection.

    Args:
        diff_thr (int): The difficulty threshold of ground truth.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    METAINFO = {
        'CLASSES':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255)]
    }

    def __init__(self,
                 diff_thr=100,
                 file_client_args: dict = dict(backend='disk'),
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.file_client_args = file_client_args
        self.file_client = FileClient(**file_client_args)
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['CLASSES'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file is None:
            img_files = glob.glob(self.data_prefix['img_path'] + '/*.png')
            for img_path in img_files:
                data_info = {}
                img_id = osp.split(self.ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['file_name'] = img_name
                data_info['img_path'] = img_path
                data_info['img_id'] = osp.split(img_path)[1]

                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes, backend='cv2')
                width, height = img.shape[:2]
                del img, img_bytes
                data_info['height'] = height
                data_info['width'] = width

                instances = []
                instance = {}
                instance['bbox'] = []
                instance['bbox_label'] = []
                instance['ignore_flag'] = 0
                instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(self.ann_file + '/*.txt')
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in ' f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + '.png'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)
                img_bytes = self.file_client.get(data_info['img_path'])
                img = mmcv.imfrombytes(img_bytes, backend='cv2')
                width, height = img.shape[:2]
                del img, img_bytes
                data_info['height'] = height
                data_info['width'] = width

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = np.array(
                            bbox_info[:8], dtype=np.float32)
                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    @property
    def bbox_min_size(self) -> Optional[str]:
        """Return the minimum size of bounding boxes in the images."""
        if self.filter_cfg is not None:
            return self.filter_cfg.get('bbox_min_size', None)
        else:
            return None

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
