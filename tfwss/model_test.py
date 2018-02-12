"""
model_test.py

The SDI Grabcut testing is done using a model trained in the ["Simple Does It" Grabcut Training for Instance Segmentation](model_train.ipynb) notebook, so make sure you've run that notebook first! We test the model on the **validation** split of the Berkeley-augmented dataset.

The Berkley augmented dataset can be downloaded from [here](
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/scaelles/OSVOS-TensorFlow/blob/master/osvos_demo.py
    Written by Sergi Caelles (scaelles@vision.ee.ethz.ch)
    This file is part of the OSVOS paper presented in:
      Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
      One-Shot Video Object Segmentation
      CVPR 2017
    Unknown code license
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import sys
import tensorflow as tf
slim = tf.contrib.slim

# Import model files
import model
from dataset import BKVOCDataset

# Parameters
gpu_id = 0
# Modify the value below to match the value of max_training_iters_3 in the training notebook!
max_training_iters = 50000

# Model paths
segnet_stream = 'weak'
ckpt_name = 'vgg_16_4chan_' + segnet_stream
ckpt_path = 'models/' + ckpt_name + '/' + ckpt_name + '.ckpt-' + str(max_training_iters)

# Load the Berkeley-augmented Pascal VOC 2012 segmentation dataset
if sys.platform.startswith("win"):
    dataset_root = "E:/datasets/bk-voc/benchmark_RELEASE/dataset"
else:
    dataset_root = '/media/EDrive/datasets/bk-voc/benchmark_RELEASE/dataset'
dataset = BKVOCDataset(phase='test', dataset_root=dataset_root)

# Display dataset configuration
dataset.print_config()

# Test the model
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        model.test(dataset, ckpt_path, dataset.pred_masks_path, dataset.img_pred_masks_path, segnet_stream)

# Combine original images with predicted instance masks
dataset.combine_images_with_predicted_masks()