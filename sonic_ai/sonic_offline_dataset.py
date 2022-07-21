import glob
import os
from pathlib import Path
import os.path as osp

import numpy as np
from mmdet.datasets.builder import DATASETS

from mmrotate.core import poly2obb_np
from mmrotate.datasets.dota import DOTADataset


@DATASETS.register_module()
class SonicOffLineDataset(DOTADataset):

    def __init__(self, category_list, *args, **kwargs):
        super().__init__(classes=category_list, *args, **kwargs)

    def load_annotations(self, ann_folder):
        """
                   Args:
                       ann_folder: folder that contains DOTA v1 annotations txt files
               """
        cls_map = {
            c: i
            for i, c in enumerate(self.CLASSES)
        }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        img_files = glob.glob(self.img_prefix + '*')
        img_files_dic = {}
        for img_file in img_files:
            img_files_dic[Path(img_file).stem] = Path(img_file).name
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                data_info['filename'] = img_files_dic[Path(ann_file).stem]
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        if cls_name not in cls_map:
                            continue
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros(
                        (0, 8), dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos
