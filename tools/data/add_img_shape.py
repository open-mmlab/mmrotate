# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
from functools import partial
from multiprocessing import Pool

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def add_parser(parser):
    """Add arguments."""
    parser.add_argument(
        '--nproc', type=int, default=10, help='the procession number')

    # argument for loading data
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='images and annotations dirs, must give a value')


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Add image shape in the end line of annotations')
    add_parser(parser)
    args = parser.parse_args()
    assert args.data_dir is not None, "argument img_dir can't be None"
    return args


def load_data(img_dir, ann_dir, save_dir, nproc=10):
    """Load dataset.

    Args:
        img_dir (str): Path of images.
        ann_dir (str): Path of annotations.
        save_dir (str): Path of new annotations.
        nproc (int): number of processes.

    Returns:
        list: Dataset's contents.
    """
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
    assert osp.isdir(ann_dir), f'The {ann_dir} is not an existing dir!'

    _load_func = partial(
        _load_data_single, img_dir=img_dir, ann_dir=ann_dir, save_dir=save_dir)
    if nproc > 1:
        pool = Pool(nproc)
        pool.map(_load_func, os.listdir(img_dir))
        pool.close()
    else:
        list(map(_load_func, os.listdir(img_dir)))


def _load_data_single(imgfile, img_dir, ann_dir, save_dir):
    """Load single image.

    Args:
        imgfile (str): Filename of single image.
        img_dir (str): Path of images.
        ann_dir (str): Path of annotations.
        save_dir (str): Path of new annotations.

    Returns:
        dict: Content of single image.
    """
    img_id, ext = osp.splitext(imgfile)
    imgpath = osp.join(img_dir, imgfile)
    size = Image.open(imgpath).size
    txtfile = osp.join(ann_dir, img_id + '.txt')
    dstfile = osp.join(save_dir, img_id + '.txt')
    shutil.copyfile(txtfile, dstfile)
    with open(dstfile, 'a') as f:
        f.write(f'{size[0]} {size[1]}\n')


def main():
    """Main function of image split."""
    args = parse_args()
    img_dir = osp.join(args.data_dir, 'images')
    ann_dir = osp.join(args.data_dir, 'labelTxt')
    save_dir = osp.join(args.data_dir, 'annfiles')
    assert not osp.exists(save_dir), \
        f'{osp.join(save_dir)} already exists'
    os.makedirs(save_dir)

    print('Processing...')
    start = time.time()
    load_data(
        img_dir=img_dir, ann_dir=ann_dir, save_dir=save_dir, nproc=args.nproc)
    stop = time.time()
    print(f'Finish in {int(stop - start)} second!!!')


if __name__ == '__main__':
    main()
