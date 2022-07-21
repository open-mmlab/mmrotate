import io
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter
from glob import glob
from pathlib import Path

import cv2
import mmcv
import numpy as np
from torch import distributed as dist
from tqdm import tqdm

from mmdet.datasets.builder import PIPELINES
from mmdet.utils import get_root_logger
from sonic_ai.pipelines.utils_labelme import copy_json_and_img, shape_to_points


@PIPELINES.register_module()
class LoadCategoryList:

    def __init__(self, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ['屏蔽']
        self.ignore_labels = ignore_labels

    def __call__(self, results, *args, **kwargs):
        label_path = results['label_path']
        if results.get('ignore_labels', False):
            ignore_labels = results['ignore_labels'] if results['ignore_labels'] is not None else self.ignore_labels
        else:
            ignore_labels = self.ignore_labels
        results['ignore_labels'] = ignore_labels

        category_map = {
            x.split(' ')[0]: x.strip().split(' ')[1].strip()
            for x in open(label_path, encoding='utf-8').readlines()
        }
        category_list = list(category_map.values())
        category_list = sorted(set(category_list), key=category_list.index)
        for ignore_label in ignore_labels:
            if ignore_label in category_list:
                category_list.remove(ignore_label)

        results['category_map'] = category_map
        results['category_list'] = category_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadPathList:

    def __call__(self, results, *args, **kwargs):
        dataset_path_list = results['dataset_path_list']

        logger = get_root_logger()
        logger.propagate = False
        path_list = []
        if isinstance(dataset_path_list, list):
            for dataset_path in dataset_path_list:
                path_list += glob(f'{dataset_path}/**/*', recursive=True)
        elif isinstance(dataset_path_list, str):
            path_list = glob(f'{dataset_path_list}/**/*', recursive=True)
        else:
            logger.error(
                f'数据集的路径为{dataset_path_list}，格式为{type(dataset_path_list)}, 请检查格式'
            )
            raise Exception('请检查数据集的路径')

        if len(path_list) == 0:
            logger.error(f"数据集的路径{dataset_path_list}下找不到数据，请检查路径填写是否正确")
            raise Exception('路径下没有数据集，请检查路径填写是否正确')

        results['path_set'] = set([str(Path(x)) for x in path_list])
        results['json_path_list'] = [
            x for x in path_list if x.endswith('.json')
        ]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadJsonDataList:

    def __call__(self, results, *args, **kwargs):
        json_path_list = results['json_path_list']
        logger = get_root_logger()
        logger.propagate = False

        disable_progressbar = False
        if dist.is_initialized() and dist.get_rank() != 0:
            disable_progressbar = True

        if platform.system() == 'Windows':
            json_data_list = [
                mmcv.load(x)
                for x in tqdm(json_path_list, disable=disable_progressbar)
            ]
        else:
            logger.info(f'使用多进程读取数据集')
            stream = sys.stdout
            if disable_progressbar:
                stream = io.StringIO()
            json_data_list = mmcv.track_parallel_progress(
                mmcv.load, json_path_list, nproc=8, file=stream)

        for i, x in enumerate(json_data_list):
            json_path = Path(json_path_list[i])
            x['imagePath'] = str(Path(json_path.parent, x['imagePath']))
            json_data_list[i]['json_path'] = json_path_list[i]

        results['json_data_list'] = json_data_list
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadLabelmeDataset:

    def __call__(self, results):
        category_map = results['category_map']
        json_path_list = results['json_path_list']
        path_set = results['path_set']
        json_data_list = results['json_data_list']
        ignore_labels = results['ignore_labels']

        error_file_path = []

        logger = get_root_logger()
        logger.propagate = False

        logger.info(f'筛选图片文件存在，筛选前：{len(json_data_list)}')
        keep_index = [
            i for i, x in enumerate(json_data_list)
            if x['imagePath'] in path_set
        ]
        json_data_list = [json_data_list[x] for x in keep_index]
        json_path_list = [json_path_list[x] for x in keep_index]
        logger.info(f'筛选图片文件存在，筛选后：{len(json_data_list)}')

        all_ann_label_list = {
            y['label']
            for x in json_data_list for y in x['shapes']
        }

        if len(set(all_ann_label_list) - set(category_map.keys())):
            logger.warning(
                f'发现未知类别：{set(all_ann_label_list) - set(category_map.keys())}')
            for path, x in zip(json_path_list, json_data_list):
                if set(shape['label'] for shape in x['shapes']) - set(
                        category_map.keys()) != set():
                    categories = set(shape['label'] for shape in x['shapes'])
                    logger.warning(f'异常样本：{categories}, {path}')

                    error_file_path.append(path)

        logger.info('筛选json')
        new_data_list = [
            x for x in json_data_list
            if all(y['label'] in category_map
                   for y in x['shapes']) and not all(
                       category_map[y['label']] in ignore_labels
                       for y in x['shapes'])
        ]
        logger.info(f'筛选后的样本数量：{len(new_data_list)}')
        logger.info(f'路径样例：{new_data_list[0]["imagePath"]}')

        results['json_data_list'] = new_data_list
        results['error_file_path'] = error_file_path
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class StatCategoryCounter:

    def __call__(self, results, *args, **kwargs):
        category_map = results['category_map']
        category_list = results['category_list']
        json_data_list = results['json_data_list']

        logger = get_root_logger()
        logger.propagate = False
        logger.info(f"\n映射表为{category_map}\n列表为{category_list}")

        category_counter = Counter(
            category_map.get(y['label'], y['label']) for x in json_data_list
            for y in x['shapes'])
        s = '\n'
        for k, v in category_counter.most_common():
            s += f'{k}\t{v}\n'
        logger.info(f'统计类别分布{s}')

        results['category_counter'] = category_counter
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SplitData:

    def __init__(
            self, start=0, end=1, seed=42, shuffle=True, key='json_path_list'):
        self.start = start
        self.end = end
        self.seed = seed
        self.shuffle = shuffle
        self.key = key

    def __call__(self, results):
        if results.get('start', False):
            start = results['start'] if results['start'] is not None else self.start
        else:
            start = self.start
        if results.get('end', False):
            end = results['end'] if results['end'] is not None else self.end
        else:
            end = self.end

        target_list = results[self.key]
        if self.shuffle:
            state = random.getstate()
            random.seed(self.seed)
            random.shuffle(target_list)
            random.setstate(state)
        n = len(target_list)
        start_index, end_index = int(start * n), int(end * n)
        target_list = target_list[start_index:end_index]
        results[self.key] = target_list
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CopyData:

    def __init__(self, times, key='json_data_list'):
        self.times = times
        self.key = key

    def __call__(self, results, *args, **kwargs):
        if results.get('times', False):
            times = results['times'] if results['times'] is not None else self.times
        else:
            times = self.times
        results[self.key] *= times

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadMutilChannelImgPathList:

    def __init__(self, channels, *args, **kwargs):
        self.channels = channels

    def __call__(self, results, *args, **kwargs):
        json_data_list = results['json_data_list']
        new_json_data_list = []

        logger = get_root_logger()
        logger.propagate = False

        logger.info(f"筛选多通道图中，筛选前{len(json_data_list)}组")
        for idx, json_data in enumerate(json_data_list):
            if len(json_data['image_path_list']) != self.channels:
                continue
            dirname = Path(json_data['imagePath']).parent
            for idx, image_path in enumerate(json_data['image_path_list']):
                json_data['image_path_list'][idx] = str(
                    Path(dirname, image_path))
            json_data['imagePath'] = json_data['image_path_list']
            new_json_data_list.append(json_data)
        logger.info(f"筛选后{len(new_json_data_list)}组")
        results['json_data_list'] = new_json_data_list
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class Labelme2Coco:

    def __call__(self, results, *args, **kwargs):
        category_list = results['category_list']
        category_map = results['category_map']
        json_data_list = results['json_data_list']
        ignore_labels = results['ignore_labels']
        error_file_path = results['error_file_path']

        logger = get_root_logger()
        logger.propagate = False

        annotations = []
        images = []
        obj_count = 0
        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            disable = False
        else:
            disable = True
        with tqdm(json_data_list, desc='labelme2coco',
                  disable=disable) as pbar:
            for idx, data in enumerate(pbar):
                filename = data['imagePath']

                height, width = data['imageHeight'], data['imageWidth']
                images.append(
                    dict(
                        id=idx, file_name=filename, height=height,
                        width=width))

                for shape in data['shapes']:
                    if shape['label'] not in category_map:
                        logger.warning('发现未知标签', idx, shape)
                        continue
                    if category_map[shape['label']] in ignore_labels:
                        continue

                    new_points = []
                    try:
                        new_points = shape_to_points(shape, height, width)
                    except:
                        logger.error(traceback.format_exc())

                    if len(new_points) == 0:
                        logger.warning(
                            f"解析 shape 失败, {shape['label']}，图片路径为{filename}")
                        error_file_path.append(data['json_path'])
                        continue

                    px = [x[0] for x in new_points]
                    py = [x[1] for x in new_points]
                    poly = new_points.flatten().tolist()
                    x_min, y_min, x_max, y_max = (
                        min(px), min(py), max(px), max(py))

                    # 处理越界的 bbox
                    x_max = min(x_max, width - 1)
                    y_max = min(y_max, height - 1)

                    category_id = category_list.index(
                        category_map[shape['label']])
                    data_anno = dict(
                        image_id=idx,
                        id=obj_count,
                        category_id=category_id,
                        bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                        area=(x_max - x_min) * (y_max - y_min),
                        segmentation=[poly],
                        iscrowd=0)

                    annotations.append(data_anno)
                    obj_count += 1

        categories = [
            {
                'id': i,
                'name': x
            } for i, x in enumerate(category_list)
        ]
        results['error_file_path'] = error_file_path
        results['json_data_dic'] = dict(
            images=images, annotations=annotations, categories=categories)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SaveJson:

    def __call__(self, results, *args, **kwargs):
        save_path = results['ann_save_path']
        json_data_dic = results['json_data_dic']

        logger = get_root_logger()
        logger.info(f'保存路径{save_path}')
        logger.info(f'样本数量：{len(json_data_dic["images"])}')
        logger.info(f'标注shape数量{len(json_data_dic["annotations"])}')
        mmcv.dump(json_data_dic, save_path)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CopyErrorPath:

    def __init__(
            self, copy_error_file_path='/data/14-调试数据/cyf', *args, **kwargs):
        self.copy_error_file_path = copy_error_file_path

    def __call__(self, results, *args, **kwargs):
        error_file_path = results['error_file_path']
        dataset_path_list = results['dataset_path_list']
        date = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        logger = get_root_logger()
        logger.propagate = False
        logger.info(f"无法用来训练的数据的长度为{len(error_file_path)}")

        dataset_commonpath = os.path.commonpath(dataset_path_list)
        for path in error_file_path:
            copy_json_and_img(
                path,
                Path(
                    self.copy_error_file_path, date,
                    Path(path).relative_to(dataset_commonpath)))
        logger = get_root_logger()
        logger.propagate = False
        logger.info(f"复制错误数据至{str(Path(self.copy_error_file_path, date))}完成")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CalculateMeanAndStd:

    def __init__(self, img_scale):
        self.img_scale = img_scale

    def __call__(self, results, *args, **kwargs):
        json_data_list = results['json_data_list']

        logger = get_root_logger()
        logger.propagate = False

        logger.info('读取图像中')
        if isinstance(json_data_list[0]['imagePath'], list):
            imgs = [
                [
                    np.resize(
                        cv2.imdecode(
                            np.fromfile(path, dtype=np.uint8),
                            cv2.IMREAD_UNCHANGED), self.img_scale)
                    for path in json_data['imagePath']
                ] for json_data in tqdm(json_data_list)
            ]
        else:
            imgs = [
                [
                    np.resize(
                        cv2.imdecode(
                            np.fromfile(
                                json_data['imagePath'], dtype=np.uint8),
                            cv2.IMREAD_UNCHANGED), self.img_scale + (3, ))
                ] for json_data in tqdm(json_data_list)
            ]

        imgs = np.array(imgs)
        mean = np.mean(imgs, axis=(0, 2, 3), where=imgs > 0)
        std = np.std(imgs, axis=(0, 2, 3), where=imgs > 0)
        logger.info("计算均值和标注差中")
        logger.info(
            f"图像列表的平均值为{[round(i,2) for i in list(mean)]}，标准差为{[round(i,2) for i in list(std)]}"
        )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
