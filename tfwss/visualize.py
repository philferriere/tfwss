"""
visualize.py

Davis 2016 visualization helpers.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/matterport/Mask_RCNN/blob/master/visualize.py
        Copyright (c) 2017 Matterport, Inc. / Written by Waleed Abdulla
        Licensed under the MIT License

References for future work:
    E:/repos/models-master/research/object_detection/utils/visualization_utils.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython.display import HTML
import io, base64
import imageio
import numpy as np
import random
import colorsys
import matplotlib.pyplot as plt

def random_colors(N, bright=True, RGB_max=255):
    """
    Generate random colors. To get visually distinct colors, generate them in HSV space then convert to RGB.
        Args:
            N: number of colors to generate.
            bright: set to True for bright colors.
            RGB_max: set to 1.0 or 255, based on image type you're working with
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [(color[0] * RGB_max, color[1] * RGB_max, color[2] * RGB_max) for color in colors]
    random.shuffle(colors)
    return colors

def display_images(images, titles=None, cols=4, cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
        Args:
            images: list or array of image tensors in HWC format.
            titles: optional. A list of titles to display with each image.
            cols: number of images per row
            cmap: Optional. Color map to use. For example, "Blues".
            norm: Optional. A Normalize instance to map values to colors.
            interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    width = 20
    plt.figure(figsize=(width, width * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        # plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
        i += 1
    plt.tight_layout()
    plt.show()

def draw_box(image, bbox, color, in_place=True):
    """Draw (in-place, or not) 3-pixel-width bounding bboxes on an image.
        Args:
            image: video frame (H,W,3)
            bbox: y1, x1, y2, x2 bounding box
            color: color list of 3 int values for RGB
            in_place: in place / copy flag
        Returns:
            image with bounding box
    """
    y1, x1, y2, x2 = bbox
    result = image if in_place == True else np.copy(image)
    result[y1:y1 + 2, x1:x2] = color
    result[y2:y2 + 2, x1:x2] = color
    result[y1:y2, x1:x1 + 2] = color
    result[y1:y2, x2:x2 + 2] = color
    return result

def draw_mask(image, mask, color, alpha=0.5, in_place=False):
    """Draw (in-place, or not) a mask on an image.
        Args:
            image: input image (H,W,3)
            mask: mask (H,W,1)
            color: color list of 3 int values for RGB
            alpha: alpha blending level
            in_place: in place / copy flag
        Returns:
            image with mask
    """
    assert(len(image.shape) == len(mask.shape) == len(color) == 3)
    assert(image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])
    threshold = (np.max(mask) - np.min(mask)) / 2
    multiplier = 1 if np.amax(color) > 1 else 255
    masked_image = image if in_place == True else np.copy(image)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask[:,:,0] > threshold,
                                         masked_image[:, :, c] *
                                         (1 - alpha) + alpha * color[c] * multiplier,
                                         masked_image[:, :, c])
    return masked_image

def draw_masks(image, bboxes, masks, alpha=0.5, in_place=True):
    """Apply the given instance masks to the image and draw their bboxes.
        Args:
            image: input image (H, W, 3)
            bboxes: (num_instances, (y1, x1, y2, x2)) bounding boxes as numpy array
            masks: masks (num_instances, H, W, 1) as numpy array
            alpha: alpha blending level
            in_place: in place / copy flag
        Returns:
            image with masks overlaid
    """
    # Number of instances
    num_instances = bboxes.shape[0]
    assert(num_instances == masks.shape[0])

    # Make a copy of the input image, if requested
    masked_image = image if in_place == True else np.copy(image)

    # Draw bboxes and masks on the image, if the bbox is not empty, using a random color
    colors = random_colors(num_instances)
    for instance in range(num_instances):
        if not np.any(bboxes[instance]):
            continue
        color = colors[instance]
        draw_mask(masked_image, masks[instance], color, alpha=alpha, in_place=True)
        # draw_mask(masked_image, masks[instance, :, :, 0], color, alpha=alpha, in_place=True)
        draw_box(masked_image, bboxes[instance], color, in_place=True)

    return masked_image
