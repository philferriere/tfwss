"""
model_train.py

This file performs training of the SDI Grabcut weakly supervised model for **instance segmentation**.
Following the instructions provided in Section "6. Instance Segmentation Results" of the "Simple Does It" paper, we use
the Berkeley-augmented Pascal VOC segmentation dataset that provides per-instance segmentation masks for VOC2012 data.

The Berkley augmented dataset can be downloaded from [here](
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

The SDI Grabcut training is done using a **4-channel input** VGG16 network pre-trained on ImageNet, so make sure to run
the [`VGG16 Surgery`](vgg16_surgery.ipynb) notebook first!

To monitor training, run:
```
# On Windows
tensorboard --logdir E:\repos\tf-wss\tfwss\models\vgg_16_4chan_weak
# On Ubuntu
tensorboard --logdir /media/EDrive/repos/tf-wss/tfwss/models/vgg_16_4chan_weak
http://<hostname>:6006
```

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/scaelles/OSVOS-TensorFlow/blob/master/osvos_parent_demo.py
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

import os
import sys
import tensorflow as tf
slim = tf.contrib.slim

# Import model files
import model
from dataset import BKVOCDataset

# Model paths
# Pre-trained VGG_16 downloaded from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
imagenet_ckpt = 'models/vgg_16_4chan/vgg_16_4chan.ckpt'
segnet_stream = 'weak'
ckpt_name = 'vgg_16_4chan_' + segnet_stream
logs_path = 'models/' + ckpt_name

# Training parameters
gpu_id = 0
iter_mean_grad = 10
max_training_iters_1 = 15000
max_training_iters_2 = 30000
max_training_iters_3 = 50000
save_step = 5000
test_image = None
display_step = 100
ini_lr = 1e-8
boundaries = [10000, 15000, 25000, 30000, 40000]
values = [ini_lr, ini_lr * 0.1, ini_lr, ini_lr * 0.1, ini_lr, ini_lr * 0.1]

# Load the Berkeley-augmented Pascal VOC 2012 segmentation dataset
if sys.platform.startswith("win"):
    dataset_root = "E:/datasets/bk-voc/benchmark_RELEASE/dataset"
else:
    dataset_root = '/media/EDrive/datasets/bk-voc/benchmark_RELEASE/dataset'
dataset = BKVOCDataset(phase='train', dataset_root=dataset_root)

# Display dataset configuration
dataset.print_config()

# Train the network with strong side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 1, learning_rate, logs_path, max_training_iters_1, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, test_image_path=test_image,
                           ckpt_name=ckpt_name)
# Train the network with weak side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(max_training_iters_1, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 2, learning_rate, logs_path, max_training_iters_2, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, resume_training=True,
                           test_image_path=test_image, ckpt_name=ckpt_name)
# Train the network without side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(max_training_iters_2, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 3, learning_rate, logs_path, max_training_iters_3, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, resume_training=True,
                           test_image_path=test_image, ckpt_name=ckpt_name)


