# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmdet.core.mask import BitmapMasks


class RBitmapMasks(BitmapMasks):

    def get_rbboxes(self):
        num_masks = len(self)
        rboxes = np.zeros((num_masks, 5), dtype=np.float32)
        x_any = self.masks.any(axis=1)
        y_any = self.masks.any(axis=2)
        for idx in range(num_masks):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                contours = cv2.findContours(self.masks[idx], cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)[0][0]
                (cx, cy), (w, h), a = cv2.minAreaRect(contours)
                rboxes[idx, :] = np.array(
                    [cx, cy, w, h, np.radians(a)], dtype=np.float32)
        return rboxes
