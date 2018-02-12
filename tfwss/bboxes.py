"""
bboxes.py

Bounding box utility functions.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/matterport/Mask_RCNN/blob/master/utils.py
        Copyright (c) 2017 Matterport, Inc. / Written by Waleed Abdulla
        Licensed under the MIT License

References for future work:
    - https://github.com/tensorflow/models/blob/master/research/object_detection/utils/np_box_ops.py
      https://github.com/tensorflow/models/blob/master/research/object_detection/utils/ops.py
        Copyright 2017 The TensorFlow Authors. All Rights Reserved.
        Licensed under the Apache License, Version 2.0
    - https://github.com/tangyuhao/DAVIS-2016-Chanllege-Solution/blob/master/Step1-SSD/tf_extended/bboxes.py
        https://github.com/tangyuhao/DAVIS-2016-Chanllege-Solution/blob/master/Step1-SSD/bounding_box.py
        Copyright (c) 2017 Paul Balanca / Written by Paul Balanca
        Licensed under the Apache License, Version 2.0, January 2004
"""

import numpy as np

def extract_bbox(mask, order='y1x1y2x2'):
    """Compute bounding box from a mask.
    Param:
        mask: [height, width]. Mask pixels are either >0 or 0.
        order: ['y1x1y2x2' | ]
    Returns:
        bbox numpy array [y1, x1, y2, x2] or tuple x1, y1, x2, y2.
    Based on:
        https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    if order == 'x1y1x2y2':
        return x1, y1, x2, y2
    else:
        return np.array([y1, x1, y2, x2])

def extract_bboxes(mask):
    """Compute bounding boxes from an array of masks.
    Params
        mask: [height, width, num_instances]. Mask pixels are either >0 or 0.
    Returns:
        bbox numpy arrays [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        boxes[i] = extract_bbox(mask[:, :, i])
    return boxes.astype(np.int32)

