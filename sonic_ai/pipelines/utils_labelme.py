import datetime
import io
import math
import os
import pathlib
import platform
import random
import shutil
import sys
import traceback
from collections import Counter
from glob import glob
from pathlib import Path

import cv2
import mmcv
import numpy as np
from torch import distributed as dist
from tqdm import tqdm

from mmdet.utils import get_root_logger


def cv_imread(file_path):
    cv_img = cv2.imdecode(
        np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


def shape_to_points(shape, height, width):
    from mmdet.utils import get_root_logger
    logger = get_root_logger()
    shape_type = shape['shape_type']
    points = shape['points']
    if shape_type == 'polygon':
        new_points = points
        if len(points) < 3:
            new_points = []
            logger.warning(f'polygon 异常，少于三个点：{shape}')
        if False in (np.array(points).max(axis=0) < [width, height]):
            new_points = []
            logger.warning(f'polygon 异常，mask 超出范围')

    elif shape_type == 'rectangle':
        (x1, y1), (x2, y2) = points
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        new_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif shape_type == "circle":
        # Create polygon shaped based on connecting lines from/to following degress
        bearing_angles = [
            0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210,
            225, 240, 255, 270, 285, 300, 315, 330, 345, 360
        ]

        orig_x1 = points[0][0]
        orig_y1 = points[0][1]

        orig_x2 = points[1][0]
        orig_y2 = points[1][1]

        # Calculate radius of circle
        radius = math.sqrt((orig_x2 - orig_x1)**2 + (orig_y2 - orig_y1)**2)

        circle_polygon = []

        for i in range(0, len(bearing_angles) - 1):
            ad1 = math.radians(bearing_angles[i])
            x1 = radius * math.cos(ad1)
            y1 = radius * math.sin(ad1)
            circle_polygon.append((orig_x1 + x1, orig_y1 + y1))

            ad2 = math.radians(bearing_angles[i + 1])
            x2 = radius * math.cos(ad2)
            y2 = radius * math.sin(ad2)
            circle_polygon.append((orig_x1 + x2, orig_y1 + y2))

        new_points = circle_polygon
    else:
        new_points = []
        logger.warning(f'未知 shape_type：{shape["shape_type"]}')

    new_points = np.asarray(new_points)
    return new_points


def labelme2coco(
        json_data_list,
        category_map,
        copy_error_file_path=None,
        dataset_commonpath=None):
    from mmdet.utils import get_root_logger
    logger = get_root_logger()
    category_list = list(category_map.values())
    category_list = sorted(set(category_list), key=category_list.index)
    if '屏蔽' in category_list:
        category_list.remove('屏蔽')
    annotations = []
    images = []
    obj_count = 0
    if (dist.is_initialized()
            and dist.get_rank() == 0) or not dist.is_initialized():
        disable = False
    else:
        disable = True
    with tqdm(json_data_list, desc='labelme2coco', disable=disable) as pbar:
        for idx, data in enumerate(pbar):
            img_filename = data['imagePath']
            height, width = data['imageHeight'], data['imageWidth']
            images.append(
                dict(
                    id=idx, file_name=img_filename, height=height,
                    width=width))

            for shape in data['shapes']:
                if shape['label'] not in category_map:
                    logger.warning('发现未知标签', idx, shape)
                    continue
                if category_map[shape['label']] == '屏蔽':
                    continue

                new_points = []
                try:
                    new_points = shape_to_points(shape, height, width)
                except:
                    logger.error(traceback.format_exc())

                if len(new_points) == 0:
                    logger.warning(
                        f'解析 shape 失败：{idx}, {shape}，图片路径为{img_filename}')
                    if copy_error_file_path:
                        date = datetime.datetime.now().strftime('%Y-%m-%d')
                        copy_json_and_img(
                            data['json_path'],
                            Path(
                                copy_error_file_path, date,
                                Path(data['json_path']).relative_to(
                                    dataset_commonpath)))
                    continue

                px = [x[0] for x in new_points]
                py = [x[1] for x in new_points]
                poly = new_points.flatten().tolist()
                x_min, y_min, x_max, y_max = (
                    min(px), min(py), max(px), max(py))

                # 处理越界的 bbox
                x_max = min(x_max, width - 1)
                y_max = min(y_max, height - 1)

                category_id = category_list.index(category_map[shape['label']])
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

    categories = [{'id': i, 'name': x} for i, x in enumerate(category_list)]
    data = dict(images=images, annotations=annotations, categories=categories)

    return data


def load_labelme_datasets(
        dataset_path_list,
        category_map,
        copy_error_file_path=None,
        dataset_commonpath=None):
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
            f'数据集的路径为{dataset_path_list}，格式为{type(dataset_path_list)}, 请检查格式')
        raise Exception('请检查数据集的路径')

    if len(path_list) == 0:
        logger.error(f"数据集的路径{dataset_path_list}下找不到数据，请检查路径填写是否正确")
        raise Exception('路径下没有数据集，请检查路径填写是否正确')

    path_list = [str(Path(x)) for x in path_list]
    json_path_list = [x for x in path_list if x.endswith('.json')]

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

    logger.info(f'筛选图片文件存在，筛选前：{len(json_data_list)}')
    path_set = set(path_list)
    keep_index = [
        i for i, x in enumerate(json_data_list) if x['imagePath'] in path_set
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
                if copy_error_file_path:
                    date = datetime.datetime.now().strftime('%Y-%m-%d')
                    copy_json_and_img(
                        path,
                        Path(
                            copy_error_file_path, date,
                            Path(path).relative_to(dataset_commonpath)))

    # logger.info(f'统计类别分布')
    # category_counter = Counter(
    #     category_map.get(y['label'], y['label']) for x in json_data_list
    #     for y in x['shapes'])
    # s = '\n'
    # for k, v in category_counter.most_common():
    #     s += f'{k}\t{v}\n'
    # logger.info(s)

    logger.info('筛选json')
    new_data_list = [
        x for x in json_data_list if all(
            y['label'] in category_map for y in x['shapes'])
    ]
    logger.info(f'筛选后的样本数量：{len(new_data_list)}')
    logger.info(f'路径样例：{new_data_list[0]["imagePath"]}')

    logger.info(f'统计类别分布')
    category_counter = Counter(
        category_map.get(y['label'], y['label']) for x in json_data_list
        for y in x['shapes'])
    s = '\n'
    for k, v in category_counter.most_common():
        s += f'{k}\t{v}\n'
    logger.info(s)

    return new_data_list


TIMEFORMAT_STAMP = '%Y%m%d_%H%M%S'


def split_data(json_data_list, start, end, seed=42, shuffle=True):
    if shuffle:
        random.seed(seed)
        random.shuffle(json_data_list)
    n = len(json_data_list)
    start_index, end_index = int(start * n), int(end * n)
    json_data_list = json_data_list[start_index:end_index]
    return json_data_list


def save_json(json_data_dic, save_path):
    from mmdet.utils import get_root_logger
    logger = get_root_logger()
    logger.info(f'保存路径{save_path}')
    logger.info(f'样本数量：{len(json_data_dic["images"])}')
    logger.info(f'标注shape数量{len(json_data_dic["annotations"])}')
    mmcv.dump(json_data_dic, save_path)


def copy_json_and_img(json_src, target_dir):
    if isinstance(json_src, list):
        for json in json_src:
            json_dst = os.path.join(target_dir, os.path.basename(json))
            copy_file(json, json_dst)
            json_data = mmcv.load(json)
            img_path = json_data['imagePath']
            img_dst = os.path.join(target_dir, img_path)
            img_src = str(pathlib.Path(json).with_name(img_path))
            copy_file(img_src, img_dst)
            if json_data.get('image_path_list', False):
                for image_path in json_data['image_path_list']:
                    dirname = Path(json).parent
                    image_dst = os.path.join(target_dir, image_path)
                    image_src = Path(dirname, image_path)
                    copy_file(image_src, image_dst)
    else:
        json_dst = os.path.join(target_dir, os.path.basename(json_src))
        copy_file(json_src, json_dst)
        img_path = mmcv.load(json_src)['imagePath']
        img_dst = os.path.join(target_dir, img_path)
        img_src = str(pathlib.Path(json_src).with_name(img_path))
        copy_file(img_src, img_dst)
        json_data = mmcv.load(json_src)
        if json_data.get('image_path_list', False):
            for image_path in json_data['image_path_list']:
                dirname = Path(json_src).parent
                image_dst = os.path.join(target_dir, image_path)
                image_src = Path(dirname, image_path)
                copy_file(image_src, image_dst)

def copy_file(src, dst):
    logger = get_root_logger()
    src = str(pathlib.Path(src))
    dst = str(pathlib.Path(dst))
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        logger.debug(f"复制 {src} 成功")
    except:
        logger.error(f"复制 {src} 失败")
        logger.error(traceback.format_exc())


def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def shape_to_bbox(shape, height, width):
    points = shape_to_points(shape, height, width)
    px = [x[0] for x in points]
    py = [x[1] for x in points]
    x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))
    x_max = min(x_max, width - 1)
    y_max = min(y_max, height - 1)

    return [x_min, y_min, x_max, y_max]
