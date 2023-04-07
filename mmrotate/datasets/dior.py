# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Optional, Union

import mmcv
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get, get_local_path, list_from_file

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DIORDataset(BaseDataset):
    """DIOR dataset for detection.

    Args:
        ann_subdir (str): Subdir where annotations are.
            Defaults to 'Annotations/Oriented Bounding Boxes/'.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        ann_type (str): Choose obb or hbb as ground truth.
            Defaults to `obb`.
    """

    METAINFO = {
        'classes':
        ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
         'chimney', 'expressway-service-area', 'expressway-toll-station',
         'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
         'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
         'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42),
                    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0),
                    (120, 166, 157)]
    }

    def __init__(self,
                 ann_subdir: str = 'Annotations/Oriented Bounding Boxes/',
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 ann_type: str = 'obb',
                 **kwargs) -> None:
        assert ann_type in ['hbb', 'obb']
        self.ann_type = ann_type
        self.ann_subdir = ann_subdir
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/dev-1.x/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            'classes in `DIORDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self.metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = f'{img_id}.jpg'
            xml_path = osp.join(self.data_root, self.ann_subdir,
                                f'{img_id}.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = file_name
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list

    @property
    def bbox_min_size(self) -> Optional[str]:
        """Return the minimum size of bounding boxes in the images."""
        if self.filter_cfg is not None:
            return self.filter_cfg.get('bbox_min_size', None)
        else:
            return None

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information, usually it includes
                `img_id`, `file_name`, and `xml_path`.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_path = osp.join(self.data_prefix['img_path'],
                            img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['xml_path'] = img_info['xml_path']

        # deal with xml file
        with get_local_path(
                img_info['xml_path'],
                backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()

        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            width, height = img.shape[:2]
            del img, img_bytes

        data_info['height'] = height
        data_info['width'] = width

        instances = []
        for obj in root.findall('object'):
            instance = {}
            cls = obj.find('name').text.lower()
            label = self.cat2label[cls]
            if label is None:
                continue

            if self.ann_type == 'obb':
                bnd_box = obj.find('robndbox')
                polygon = np.array([
                    float(bnd_box.find('x_left_top').text),
                    float(bnd_box.find('y_left_top').text),
                    float(bnd_box.find('x_right_top').text),
                    float(bnd_box.find('y_right_top').text),
                    float(bnd_box.find('x_right_bottom').text),
                    float(bnd_box.find('y_right_bottom').text),
                    float(bnd_box.find('x_left_bottom').text),
                    float(bnd_box.find('y_left_bottom').text),
                ]).astype(np.float32)
            else:
                bnd_box = obj.find('bndbox')
                if bnd_box is None:
                    continue
                polygon = np.array([
                    float(bnd_box.find('xmin').text),
                    float(bnd_box.find('ymin').text),
                    float(bnd_box.find('xmax').text),
                    float(bnd_box.find('ymin').text),
                    float(bnd_box.find('xmax').text),
                    float(bnd_box.find('ymax').text),
                    float(bnd_box.find('xmin').text),
                    float(bnd_box.find('ymax').text)
                ]).astype(np.float32)

            ignore = False
            if self.bbox_min_size is not None:
                assert not self.test_mode
                if width < self.bbox_min_size or height < self.bbox_min_size:
                    ignore = True
            if ignore:
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = polygon
            instance['bbox_label'] = label
            instances.append(instance)

        data_info['instances'] = instances
        return data_info

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
        """Get DIOR category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
