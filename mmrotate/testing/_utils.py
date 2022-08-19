# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os.path import dirname, exists, join

import numpy as np
import torch
<<<<<<< HEAD
<<<<<<< HEAD
from mmdet.structures import DetDataSample
from mmengine.dataset import pseudo_collate
from mmengine.structures import InstanceData, PixelData
<<<<<<< HEAD

from mmrotate.core.bbox.structures import RotatedBoxes

=======
=======
from mmdet.structures import DetDataSample
>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
from mmengine.data import BaseDataElement as PixelData
from mmengine.data import InstanceData
=======
>>>>>>> c95ac9c (fix lint)

from mmrotate.core.bbox.structures import RotatedBoxes

<<<<<<< HEAD
from mmdet.structures import DetDataSample
>>>>>>> 61dcdf7 (init)
=======
>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)

def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmrotate repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
<<<<<<< HEAD
<<<<<<< HEAD
        import mmrotate
=======
        import mmdet
>>>>>>> 61dcdf7 (init)
=======
        import mmrotate
>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
        repo_dpath = dirname(dirname(mmrotate.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmengine import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 61dcdf7 (init)
=======

>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
def _rand_bboxes(rng, num_boxes, w, h):
    cx, cy, bw, bh, t = rng.rand(num_boxes, 5).T
    bboxes = np.vstack([cx * w, cy * h, w * bw, h * bh, t]).T
    return bboxes

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 61dcdf7 (init)
=======

>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
def _rand_masks(rng, num_boxes, bboxes, img_w, img_h):
    from mmdet.structures.mask import BitmapMasks
    masks = np.zeros((num_boxes, img_h, img_w))
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        mask = (rng.rand(1, bbox[3] - bbox[1], bbox[2] - bbox[0]) >
                0.3).astype(np.int)
        masks[i:i + 1, bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask
    return BitmapMasks(masks, height=img_h, width=img_w)

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 61dcdf7 (init)
=======

>>>>>>> 9ce6d05 (fix lint & modify 'oc' in RotatedBoxes)
def demo_mm_inputs(batch_size=2,
                   image_shapes=(3, 128, 128),
                   num_items=None,
                   num_classes=10,
                   sem_seg_output_strides=1,
                   with_mask=False,
<<<<<<< HEAD
<<<<<<< HEAD
                   with_semantic=False,
                   with_boxlist=False,
                   device='cpu'):
=======
                   with_semantic=False):
>>>>>>> 61dcdf7 (init)
=======
                   with_semantic=False,
                   with_boxlist=False,
                   device='cpu'):
>>>>>>> c95ac9c (fix lint)
    """Create a superset of inputs needed to run test or train batches.

    Args:
        batch_size (int): batch size. Defaults to 2.
        image_shapes (List[tuple], Optional): image shape.
            Defaults to (3, 128, 128)
        num_items (None | List[int]): specifies the number
            of boxes in each batch item. Default to None.
        num_classes (int): number of different labels a
            box might have. Defaults to 10.
        with_mask (bool): Whether to return mask annotation.
            Defaults to False.
        with_semantic (bool): whether to return semantic.
            Defaults to False.
<<<<<<< HEAD
<<<<<<< HEAD
        device (str): Destination device type. Defaults to cpu.
=======
>>>>>>> 61dcdf7 (init)
=======
        device (str): Destination device type. Defaults to cpu.
>>>>>>> c95ac9c (fix lint)
    """
    rng = np.random.RandomState(0)

    if isinstance(image_shapes, list):
        assert len(image_shapes) == batch_size
    else:
        image_shapes = [image_shapes] * batch_size

    if isinstance(num_items, list):
        assert len(num_items) == batch_size

    packed_inputs = []
    for idx in range(batch_size):
        image_shape = image_shapes[idx]
        c, h, w = image_shape

        image = rng.randint(0, 255, size=image_shape, dtype=np.uint8)

        mm_inputs = dict()
<<<<<<< HEAD
<<<<<<< HEAD
        mm_inputs['inputs'] = torch.from_numpy(image).to(device)
=======
        mm_inputs['inputs'] = torch.from_numpy(image)
>>>>>>> 61dcdf7 (init)
=======
        mm_inputs['inputs'] = torch.from_numpy(image).to(device)
>>>>>>> c95ac9c (fix lint)

        img_meta = {
            'img_id': idx,
            'img_shape': image_shape[1:],
            'ori_shape': image_shape[1:],
            'filename': '<demo>.png',
            'scale_factor': np.array([1.1, 1.2]),
            'flip': False,
            'flip_direction': None,
            'border': [1, 1, 1, 1]  # Only used by CenterNet
        }

        data_sample = DetDataSample()
        data_sample.set_metainfo(img_meta)

        # gt_instances
        gt_instances = InstanceData()
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[idx]

        bboxes = _rand_bboxes(rng, num_boxes, w, h)
        labels = rng.randint(1, num_classes, size=num_boxes)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c95ac9c (fix lint)
        # TODO: remove this part when all model adapted with BaseBoxes
        if with_boxlist:
            gt_instances.bboxes = RotatedBoxes(bboxes, dtype=torch.float32)
        else:
            gt_instances.bboxes = torch.FloatTensor(bboxes)
<<<<<<< HEAD
=======
        gt_instances.bboxes = RotatedBoxes(torch.FloatTensor(bboxes))
>>>>>>> 61dcdf7 (init)
=======
>>>>>>> c95ac9c (fix lint)
        gt_instances.labels = torch.LongTensor(labels)

        if with_mask:
            masks = _rand_masks(rng, num_boxes, bboxes, w, h)
            gt_instances.masks = masks

        # TODO: waiting for ci to be fixed
        # masks = np.random.randint(0, 2, (len(bboxes), h, w), dtype=np.uint8)
        # gt_instances.mask = BitmapMasks(masks, h, w)

        data_sample.gt_instances = gt_instances

        # ignore_instances
        ignore_instances = InstanceData()
        bboxes = _rand_bboxes(rng, num_boxes, w, h)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c95ac9c (fix lint)
        if with_boxlist:
            ignore_instances.bboxes = RotatedBoxes(bboxes, dtype=torch.float32)
        else:
            ignore_instances.bboxes = torch.FloatTensor(bboxes)
<<<<<<< HEAD
=======
        ignore_instances.bboxes = torch.FloatTensor(bboxes)
>>>>>>> 61dcdf7 (init)
=======
>>>>>>> c95ac9c (fix lint)
        data_sample.ignored_instances = ignore_instances

        # gt_sem_seg
        if with_semantic:
            # assume gt_semantic_seg using scale 1/8 of the img
            gt_semantic_seg = torch.from_numpy(
                np.random.randint(
                    0,
                    num_classes, (1, h // sem_seg_output_strides,
                                  w // sem_seg_output_strides),
                    dtype=np.uint8))
            gt_sem_seg_data = dict(sem_seg=gt_semantic_seg)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

<<<<<<< HEAD
<<<<<<< HEAD
        mm_inputs['data_samples'] = data_sample.to(device)
=======
        mm_inputs['data_sample'] = data_sample
>>>>>>> 61dcdf7 (init)
=======
        mm_inputs['data_samples'] = data_sample.to(device)
>>>>>>> c95ac9c (fix lint)

        # TODO: gt_ignore

        packed_inputs.append(mm_inputs)
<<<<<<< HEAD
<<<<<<< HEAD
    data = pseudo_collate(packed_inputs)
    return data
=======
    return packed_inputs
>>>>>>> 61dcdf7 (init)
=======
    data = pseudo_collate(packed_inputs)
    return data
>>>>>>> c95ac9c (fix lint)
