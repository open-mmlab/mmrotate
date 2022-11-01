# Copyright (c) OpenMMLab. All rights reserved.
# Written by dingjiansw101
# Reference: https://github.com/CAPTAIN-WHU/DOTA_devkit
import json
import os
from argparse import ArgumentParser

import cv2
from mmengine.utils import ProgressBar

try:
    import shapely.geometry as shgeo
except ImportError:
    shgeo = None

wordname_15 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('srcpath', help='Source path')
    parser.add_argument('destpath', help='destination path')
    args = parser.parse_args()
    return args


def parse_dota_poly(filename):
    """parse the dota ground truth in the format:

    [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    f = []
    fd = open(filename, 'r')
    f = fd
    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            # clear the wrong name after check all the data
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                object_struct['difficult'] = splitlines[9]
            object_struct['poly'] = [
                (float(splitlines[0]), float(splitlines[1])),
                (float(splitlines[2]), float(splitlines[3])),
                (float(splitlines[4]), float(splitlines[5])),
                (float(splitlines[6]), float(splitlines[7]))
            ]
            if shgeo is None:
                raise ImportError('Please run "pip install shapely" '
                                  'to install shapely first.')
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            objects.append(object_struct)
        else:
            break
    return objects


def parse_dota_poly2(filename):
    """parse the dota ground truth in the format:

    [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    objects = parse_dota_poly(filename)
    for obj in objects:
        poly = obj['poly']
        obj['poly'] = [
            poly[0][0], poly[0][1], poly[1][0], poly[1][1], poly[2][0],
            poly[2][1], poly[3][0], poly[3][1]
        ]
        obj['poly'] = list(map(int, obj['poly']))
    return objects


def main(args):
    imageparent = os.path.join(args.srcpath, 'images')
    labelparent = os.path.join(args.srcpath, 'annfiles')

    data_dict = {}
    info = {
        'contributor': 'captain group',
        'data_created': '2018',
        'description': 'This is 1.0 version of DOTA dataset.',
        'url': 'http://captain.whu.edu.cn/DOTAweb/',
        'version': '1.0',
        'year': 2018
    }
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_15):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(args.destpath, 'w') as f_out:
        filenames = []
        for root, dirs, files in os.walk(labelparent):
            for filespath in files:
                filepath = os.path.join(root, filespath)
                filenames.append(filepath)
        prog_bar = ProgressBar(len(filenames))
        for file in filenames:
            basename = os.path.basename(os.path.splitext(file)[0])
            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = parse_dota_poly2(file)
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = wordname_15.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = (min(obj['poly'][0::2]),
                                          min(obj['poly'][1::2]),
                                          max(obj['poly'][0::2]),
                                          max(obj['poly'][1::2]))

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
            prog_bar.update()
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    args = parse_args()
    main(args)
