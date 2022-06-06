# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import print_log
from mmdet.datasets import CustomDataset
from PIL import Image

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class HRSCDataset(CustomDataset):
    """HRSC dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        classwise (bool): Whether to use all classes or only ship.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    CLASSES = None
    HRSC_CLASS = ('ship', )
    HRSC_CLASSES = ('ship', 'aircraft carrier', 'warcraft', 'merchant ship',
                    'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
                    'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty Hawk',
                    'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue Ridge',
                    'Container', 'OXo|--)', 'Car carrier([]==[])',
                    'Hovercraft', 'yacht', 'CntShip(_|.--.--|_]=', 'Cruise',
                    'submarine', 'lute', 'Medical', 'Car carrier(======|',
                    'Ford-class', 'Midway-class', 'Invincible-class')
    HRSC_CLASSES_ID = ('01', '02', '03', '04', '05', '06', '07', '08', '09',
                       '10', '11', '12', '13', '14', '15', '16', '17', '18',
                       '19', '20', '22', '24', '25', '26', '27', '28', '29',
                       '30', '31', '32', '33')
    PALETTE = [
        (0, 255, 0),
    ]
    CLASSWISE_PALETTE = [(220, 20, 60), (119, 11, 32),
                         (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100),
                         (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
                         (100, 170, 30), (220, 220, 0), (175, 116, 175),
                         (250, 0, 30), (165, 42, 42), (255, 77, 255),
                         (0, 226, 252), (182, 182, 255), (0, 82, 0),
                         (120, 166, 157), (110, 76, 0), (174, 57, 255),
                         (199, 100, 0), (72, 0, 118), (255, 179, 240),
                         (0, 125, 92), (209, 0, 151), (188, 208, 182),
                         (0, 220, 176), (255, 99, 164), (92, 0, 73)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 classwise=False,
                 version='oc',
                 **kwargs):
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.classwise = classwise
        self.version = version
        if self.classwise:
            HRSCDataset.PALETTE = HRSCDataset.CLASSWISE_PALETTE
            HRSCDataset.CLASSES = self.HRSC_CLASSES
            self.catid2label = {
                ('1' + '0' * 6 + cls_id): i
                for i, cls_id in enumerate(self.HRSC_CLASSES_ID)
            }
        else:
            HRSCDataset.CLASSES = self.HRSC_CLASS
        # self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        super(HRSCDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of Imageset file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            data_info = {}

            filename = osp.join(self.img_subdir, f'{img_id}.bmp')
            data_info['filename'] = f'{img_id}.bmp'
            xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = int(root.find('Img_SizeWidth').text)
            height = int(root.find('Img_SizeHeight').text)

            if width is None or height is None:
                img_path = osp.join(self.img_prefix, filename)
                img = Image.open(img_path)
                width, height = img.size
            data_info['width'] = width
            data_info['height'] = height
            data_info['ann'] = {}
            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_headers = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            gt_polygons_ignore = []
            gt_headers_ignore = []

            for obj in root.findall('HRSC_Objects/HRSC_Object'):
                if self.classwise:
                    class_id = obj.find('Class_ID').text
                    label = self.catid2label.get(class_id)
                    if label is None:
                        continue
                else:
                    label = 0

                # Add an extra score to use obb2poly_np
                bbox = np.array([[
                    float(obj.find('mbox_cx').text),
                    float(obj.find('mbox_cy').text),
                    float(obj.find('mbox_w').text),
                    float(obj.find('mbox_h').text),
                    float(obj.find('mbox_ang').text), 0
                ]],
                                dtype=np.float32)

                polygon = obb2poly_np(bbox, 'le90')[0, :-1].astype(np.float32)
                if self.version != 'le90':
                    bbox = np.array(
                        poly2obb_np(polygon, self.version), dtype=np.float32)
                else:
                    bbox = bbox[0, :-1]
                head = np.array([
                    int(obj.find('header_x').text),
                    int(obj.find('header_y').text)
                ],
                                dtype=np.int64)

                gt_bboxes.append(bbox)
                gt_labels.append(label)
                gt_polygons.append(polygon)
                gt_headers.append(head)

            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(
                    gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(
                    gt_polygons, dtype=np.float32)
                data_info['ann']['headers'] = np.array(
                    gt_headers, dtype=np.int64)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8),
                                                        dtype=np.float32)
                data_info['ann']['headers'] = np.zeros((0, 2),
                                                       dtype=np.float32)

            if gt_polygons_ignore:
                data_info['ann']['bboxes_ignore'] = np.array(
                    gt_bboxes_ignore, dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    gt_labels_ignore, dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.array(
                    gt_polygons_ignore, dtype=np.float32)
                data_info['ann']['headers_ignore'] = np.array(
                    gt_headers_ignore, dtype=np.float32)
            else:
                data_info['ann']['bboxes_ignore'] = np.zeros((0, 5),
                                                             dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array([],
                                                             dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros(
                    (0, 8), dtype=np.float32)
                data_info['ann']['headers_ignore'] = np.zeros((0, 2),
                                                              dtype=np.float32)

            data_infos.append(data_info)
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def evaluate(
            self,
            results,
            metric='mAP',
            logger=None,
            proposal_nums=(100, 300, 1000),
            iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            scale_ranges=None,
            use_07_metric=True,
            nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            use_07_metric (bool): Whether to use the voc07 metric.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            raise NotImplementedError

        return eval_results
