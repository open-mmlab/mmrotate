# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
# import os.path as osp
import xml.etree.ElementTree as ET

import cv2 as cv
import mmengine
import mmcv
import numpy as np

# from mmdet.core import table_classes

table_classes = ('table', )

label_ids = {name: i for i, name in enumerate(table_classes)}


def parse_xml(args):
    xml_path, img_path, valid_img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    bboxes = []
    labels = []
    # bboxes_ignore = []
    # labels_ignore = []

    image = mmcv.imread(img_path)

    w = image.shape[1]
    h = image.shape[0]

    # bboxes = np.array([[0,0,200,200]], ndmin=2) - 1

    tables = root.findall('table')

    for obj in tables:
        points_str = obj.find('Coords').attrib['points']
        label = 1

        points_str_list = points_str.split(' ')
        points_list = []

        for item in points_str_list:
            x, y = item.split(',')
            x, y = int(x), int(y)
            points_list.append((x, y))

        # box_w = points_list[2][0] - points_list[0][0]
        # box_h = points_list[2][1] - points_list[0][1]

        # difficult = int(obj.find('difficult').text)

        bbox = [
            points_list[0][0], points_list[0][1], points_list[2][0],
            points_list[2][1]
        ]

        # opencv验证
        # cv.rectangle(
        #     image,
        #     points_list[0],
        #     points_list[2],
        #     color=(0, 0, 255),
        #     thickness=5)

        # cv.addText(image, 'A', points_list[0], 'FONT_HERSHEY_SIMPLEX', 1)
        # cv.addText(image, 'B', points_list[1], 'FONT_HERSHEY_SIMPLEX', 1)
        # cv.addText(image, 'C', points_list[2], 'FONT_HERSHEY_SIMPLEX', 1)
        # cv.addText(image, 'D', points_list[3], 'FONT_HERSHEY_SIMPLEX', 1)
        # cv.putText(
        #     image,
        #     xml_path,
        #     (100,100),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0,0,255),
        #     1
        # )

        bboxes.append(bbox)
        labels.append(label)

    # bboxes.append([200,200,200,100])

    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    # if not bboxes_ignore:
    #     bboxes_ignore = np.zeros((0, 4))
    #     labels_ignore = np.zeros((0, ))
    # else:
    #     bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
    #     labels_ignore = np.array(labels_ignore)

    # mmcv.imshow_bboxes(image, bboxes, colors='red', thickness=5,
    # wait_time=500)

    # mmcv.imshow(image)

    # mmcv.imwrite(image, valid_img_path)

    annotation = {
        # 'filename': img_path,
        'filename': img_path.split('/')[len(img_path.split('/')) - 1],
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            # 'bboxes_ignore': bboxes_ignore.astype(np.float32),
            # 'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(xml_dir, out_ann_file, out_valid_img_dir):

    annotations = []

    xmls = os.listdir(xml_dir)

    xml_paths = []
    img_paths = []
    valid_img_paths = []

    for item in xmls:
        # if '.xml' in item:
        if '.xml' in item and 't10' in item:

            valid_img_paths.append(out_valid_img_dir + '/' +
                                   item.replace('.xml', '.jpg'))

            xml_paths.append(xml_dir + '/' + item)

            jpg_item_jpg = item.replace('.xml', '.jpg')
            jpg_item_JPN = item.replace('.xml', '.JPG')
            jpg_item_png = item.replace('.xml', '.png')
            jpg_item_PNG = item.replace('.xml', '.PNG')
            jpg_item_TIFF = item.replace('.xml', '.TIFF')

            if jpg_item_jpg in xmls:
                img_paths.append(xml_dir + '/' + jpg_item_jpg)
            elif jpg_item_JPN in xmls:
                img_paths.append(xml_dir + '/' + jpg_item_JPN)
            elif jpg_item_png in xmls:
                img_paths.append(xml_dir + '/' + jpg_item_png)
            elif jpg_item_PNG in xmls:
                img_paths.append(xml_dir + '/' + jpg_item_PNG)
            elif jpg_item_TIFF in xmls:
                img_paths.append(xml_dir + '/' + jpg_item_TIFF)
            else:
                print(item)
                break

    part_annotations = mmengine.track_progress(
        parse_xml, list(zip(xml_paths, img_paths, valid_img_paths)))
    annotations.extend(part_annotations)

    if out_ann_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    mmengine.dump(annotations, out_ann_file, file_format='json', indent=4)
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))

        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))

        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))



        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    # for category_id, name in enumerate(table_classes()):
    #     category_item = dict()
    #     category_item['supercategory'] = str('none')
    #     category_item['id'] = int(category_id)
    #     category_item['name'] = str(name)
    #     coco['categories'].append(category_item)

    coco['categories'] = [{'id': 1, 'supercategory': 'table', 'name': 'table'}]

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        # bboxes_ignore = ann['bboxes_ignore'][:, :4]
        # labels_ignore = ann['labels_ignore']
        # for bbox_id in range(len(bboxes_ignore)):
        #     bbox = bboxes_ignore[bbox_id]
        #     label = labels_ignore[bbox_id]
        #     annotation_id = addAnnItem(
        #         annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--out-format',
        default='pkl',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def main():

    prefix = 'data/ICDAR2019_cTDaR'
    xml_path = prefix + '/training/TRACKA/ground_truth'

    json_out_path = prefix + '/train.json'
    ann = cvt_annotations(
        xml_path,
        json_out_path,
        out_valid_img_dir=prefix+'/valid_icdar2019_train')


    # test 数据集
    prefix = 'data/ICDAR2019_cTDaR'
    xml_path = prefix + '/test/TRACKA'
    json_out_path = prefix + '/test.json'
    cvt_annotations(
        xml_path,
        json_out_path,
        out_valid_img_dir=prefix+'/valid_icdar2019_test')


if __name__ == '__main__':
    main()
