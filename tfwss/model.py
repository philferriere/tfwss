"""
model.py

Segmentation backbone networks.

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

References for future work:
    https://github.com/scaelles/OSVOS-TensorFlow
    http://localhost:8889/notebooks/models-master/research/slim/slim_walkthrough.ipynb
    https://github.com/bryanyzhu/two-stream-pytorch
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    https://github.com/kwotsin/TensorFlow-ENet/blob/master/predict_segmentation.py
    https://github.com/fperazzi/davis-2017/tree/master/python/lib/davis/measures
    https://github.com/suyogduttjain/fusionseg
    https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, warnings
import numpy as np
from datetime import datetime
from skimage.io import imsave

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
slim = tf.contrib.slim

from tqdm import trange

def backbone_arg_scope(weight_decay=0.0002):
    """Defines the network's arg scope.
    Args:
        weight_decay: The l2 regularization coefficient.
    Returns:
        An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        padding='SAME') as arg_sc:
        return arg_sc


def crop_features(feature, out_size):
    """Crop the center of a feature map
    This is necessary when large upsampling results in a (width x height) size larger than the original input.
    Args:
        feature: Feature map to crop
        out_size: Size of the output feature map
    Returns:
        Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])


def backbone(inputs, segnet_stream='weak'):
    """Defines the backbone network (same as the OSVOS network, with variation in input size)
    Args:
        inputs: Tensorflow placeholder that contains the input image (either 3 or 4 channels)
        segnet_stream: Is this the 3-channel or the 4-channel input version?
    Returns:
        net: Output Tensor of the network
        end_points: Dictionary with all Tensors of the network
    Reminder:
        This is how a VGG16 network looks like:

        Layer (type)                     Output Shape          Param #     Connected to
        ====================================================================================================
        input_1 (InputLayer)             (None, 480, 854, 3)   0
        ____________________________________________________________________________________________________
        block1_conv1 (Convolution2D)     (None, 480, 854, 64)  1792        input_1[0][0]
        ____________________________________________________________________________________________________
        block1_conv2 (Convolution2D)     (None, 480, 854, 64)  36928       block1_conv1[0][0]
        ____________________________________________________________________________________________________
        block1_pool (MaxPooling2D)       (None, 240, 427, 64)  0           block1_conv2[0][0]
        ____________________________________________________________________________________________________
        block2_conv1 (Convolution2D)     (None, 240, 427, 128) 73856       block1_pool[0][0]
        ____________________________________________________________________________________________________
        block2_conv2 (Convolution2D)     (None, 240, 427, 128) 147584      block2_conv1[0][0]
        ____________________________________________________________________________________________________
        block2_pool (MaxPooling2D)       (None, 120, 214, 128) 0           block2_conv2[0][0]
        ____________________________________________________________________________________________________
        block3_conv1 (Convolution2D)     (None, 120, 214, 256) 295168      block2_pool[0][0]
        ____________________________________________________________________________________________________
        block3_conv2 (Convolution2D)     (None, 120, 214, 256) 590080      block3_conv1[0][0]
        ____________________________________________________________________________________________________
        block3_conv3 (Convolution2D)     (None, 120, 214, 256) 590080      block3_conv2[0][0]
        ____________________________________________________________________________________________________
        block3_conv4 (Convolution2D)     (None, 120, 214, 256) 590080      block3_conv3[0][0]
        ____________________________________________________________________________________________________
        block3_pool (MaxPooling2D)       (None, 60, 107, 256)  0           block3_conv4[0][0]
        ____________________________________________________________________________________________________
        block4_conv1 (Convolution2D)     (None, 60, 107, 512)  1180160     block3_pool[0][0]
        ____________________________________________________________________________________________________
        block4_conv2 (Convolution2D)     (None, 60, 107, 512)  2359808     block4_conv1[0][0]
        ____________________________________________________________________________________________________
        block4_conv3 (Convolution2D)     (None, 60, 107, 512)  2359808     block4_conv2[0][0]
        ____________________________________________________________________________________________________
        block4_conv4 (Convolution2D)     (None, 60, 107, 512)  2359808     block4_conv3[0][0]
        ____________________________________________________________________________________________________
        block4_pool (MaxPooling2D)       (None, 30, 54, 512)   0           block4_conv4[0][0]
        ____________________________________________________________________________________________________
        block5_conv1 (Convolution2D)     (None, 30, 54, 512)   2359808     block4_pool[0][0]
        ____________________________________________________________________________________________________
        block5_conv2 (Convolution2D)     (None, 30, 54, 512)   2359808     block5_conv1[0][0]
        ____________________________________________________________________________________________________
        block5_conv3 (Convolution2D)     (None, 30, 54, 512)   2359808     block5_conv2[0][0]
        ____________________________________________________________________________________________________
        block5_conv4 (Convolution2D)     (None, 30, 54, 512)   2359808     block5_conv3[0][0]
        ____________________________________________________________________________________________________
        block5_pool (MaxPooling2D)       (None, 15, 27, 512)   0           block5_conv4[0][0]
        ____________________________________________________________________________________________________
        flatten (Flatten)                (None, 207360)        0           block5_pool[0][0]
        ____________________________________________________________________________________________________
        fc1 (Dense)                      (None, 4096)          xxx         flatten[0][0]
        ____________________________________________________________________________________________________
        fc2 (Dense)                      (None, 4096)          yyy         fc1[0][0]
        ____________________________________________________________________________________________________
        predictions (Dense)              (None, 1000)          zzz         fc2[0][0]
        ====================================================================================================
    Original Code:
        ETH Zurich
    """
    im_size = tf.shape(inputs)

    with tf.variable_scope(segnet_stream, segnet_stream, [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs of all intermediate layers.
        # Make sure convolution and max-pooling layers use SAME padding by default
        # Also, group all end points in the same container/collection
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            outputs_collections=end_points_collection):

            # VGG16 stage 1 has 2 convolution blocks followed by max-pooling
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            # VGG16 stage 2 has 2 convolution blocks followed by max-pooling
            net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net_2, [2, 2], scope='pool2')

            # VGG16 stage 3 has 3 convolution blocks followed by max-pooling
            net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net_3, [2, 2], scope='pool3')

            # VGG16 stage 4 has 3 convolution blocks followed by max-pooling
            net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net_4, [2, 2], scope='pool4')

            # VGG16 stage 5 has 3 convolution blocks...
            net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # ...but here, it is not followed by max-pooling, as in the original VGG16 architecture.

            # This is where the specialization of the VGG network takes place, as described in DRIU and
            # OSVOS-S. The idea is to extract *side feature maps* and design *specialized layers* to perform
            # *deep supervision* targeted at a different task (here, segmentation) than the one used to
            # train the base network originally (i.e., large-scale natural image classification).

            # As explained in DRIU, each specialized side output produces feature maps in 16 different channels,
            # which are resized to the original image size and concatenated, creating a volume of fine-to-coarse
            # feature maps. one last convolutional layer linearly combines the feature maps from the volume
            # created by the specialized side outputs into a regressed result.  The convolutional layers employ
            # 3 x 3 convolutional filters for efficiency, except the ones used for linearly combining the outputs
            # (1 x 1 filters).

            with slim.arg_scope([slim.conv2d], activation_fn=None):

                # Convolve last layer of stage 2 (before max-pooling) -> side_2 (None, 240, 427, 16)
                side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')

                # Convolve last layer of stage 3 (before max-pooling) -> side_3 (None, 120, 214, 16)
                side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')

                # Convolve last layer of stage 4 (before max-pooling) -> side_3 (None, 60, 117, 16)
                side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')

                # Convolve last layer of stage 3 (before max-pooling) -> side_3 (None, 30, 54, 16)
                side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')

                # The _S layears are the side output that will be used for deep supervision

                # Dim reduction - linearly combine side_2 feature maps -> side_2_s (None, 240, 427, 1)
                side_2_s = slim.conv2d(side_2, 1, [1, 1], scope='score-dsn_2')

                # Dim reduction - linearly combine side_3 feature maps -> side_3_s (None, 120, 214, 1)
                side_3_s = slim.conv2d(side_3, 1, [1, 1], scope='score-dsn_3')

                # Dim reduction - linearly combine side_4 feature maps -> side_4_s (None, 60, 117, 1)
                side_4_s = slim.conv2d(side_4, 1, [1, 1], scope='score-dsn_4')

                # Dim reduction - linearly combine side_5 feature maps -> side_5_s (None, 30, 54, 1)
                side_5_s = slim.conv2d(side_5, 1, [1, 1], scope='score-dsn_5')

                # As repeated in OSVOS-S, upscaling operations take place wherever necessary, and feature
                # maps from the separate paths are concatenated to construct a volume with information from
                # different levels of detail. We linearly fuse the feature maps to a single output which has
                # the same dimensions as the input image.
                with slim.arg_scope([slim.convolution2d_transpose],
                                    activation_fn=None, biases_initializer=None, padding='VALID',
                                    outputs_collections=end_points_collection, trainable=False):

                    # Upsample the side outputs for deep supervision and center-cop them to the same size as
                    # the input. Note that this is straight upsampling (we're not trying to learn upsampling
                    # filters), hence the trainable=False param.

                    # Upsample side_2_s (None, 240, 427, 1) -> (None, 480, 854, 1)
                    # Center-crop (None, 480, 854, 1) to original image size (None, 480, 854, 1)
                    side_2_s = slim.convolution2d_transpose(side_2_s, 1, 4, 2, scope='score-dsn_2-up')
                    side_2_s = crop_features(side_2_s, im_size)
                    utils.collect_named_outputs(end_points_collection, segnet_stream + '/score-dsn_2-cr', side_2_s)

                    # Upsample side_3_s (None, 120, 214, 1) -> (None, 484, 860, 1)
                    # Center-crop (None, 484, 860, 1) to original image size (None, 480, 854, 1)
                    side_3_s = slim.convolution2d_transpose(side_3_s, 1, 8, 4, scope='score-dsn_3-up')
                    side_3_s = crop_features(side_3_s, im_size)
                    utils.collect_named_outputs(end_points_collection, segnet_stream + '/score-dsn_3-cr', side_3_s)

                    # Upsample side_4_s (None, 60, 117, 1) -> (None, 488, 864, 1)
                    # Center-crop (None, 488, 864, 1) to original image size (None, 480, 854, 1)
                    side_4_s = slim.convolution2d_transpose(side_4_s, 1, 16, 8, scope='score-dsn_4-up')
                    side_4_s = crop_features(side_4_s, im_size)
                    utils.collect_named_outputs(end_points_collection, segnet_stream + '/score-dsn_4-cr', side_4_s)

                    # Upsample side_5_s (None, 30, 54, 1) -> (None, 496, 880, 1)
                    # Center-crop (None, 496, 880, 1) to original image size (None, 480, 854, 1)
                    side_5_s = slim.convolution2d_transpose(side_5_s, 1, 32, 16, scope='score-dsn_5-up')
                    side_5_s = crop_features(side_5_s, im_size)
                    utils.collect_named_outputs(end_points_collection, segnet_stream + '/score-dsn_5-cr', side_5_s)

                    # Upsample the main outputs and center-cop them to the same size as the input
                    # Note that this is straight upsampling (we're not trying to learn upsampling filters),
                    # hence the trainable=False param. Then, concatenate thm in a big volume of fine-to-coarse
                    # feature maps of the same size.

                    # Upsample side_2 (None, 240, 427, 16) -> side_2_f (None, 480, 854, 16)
                    # Center-crop (None, 480, 854, 16) to original image size (None, 480, 854, 16)
                    side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                    side_2_f = crop_features(side_2_f, im_size)
                    utils.collect_named_outputs(end_points_collection, segnet_stream + '/side-multi2-cr', side_2_f)

                    # Upsample side_2 (None, 120, 214, 16) -> side_2_f (None, 488, 864, 16)
                    # Center-crop (None, 488, 864, 16) to original image size (None, 480, 854, 16)
                    side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                    side_3_f = crop_features(side_3_f, im_size)
                    utils.collect_named_outputs(end_points_collection, segnet_stream + '/side-multi3-cr', side_3_f)

                    # Upsample side_2 (None, 60, 117, 16) -> side_2_f (None, 488, 864, 16)
                    # Center-crop (None, 488, 864, 16) to original image size (None, 480, 854, 16)
                    side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                    side_4_f = crop_features(side_4_f, im_size)
                    utils.collect_named_outputs(end_points_collection, segnet_stream + '/side-multi4-cr', side_4_f)

                    # Upsample side_2 (None, 30, 54, 16) -> side_2_f (None, 496, 880, 16)
                    # Center-crop (None, 496, 880, 16) to original image size (None, 480, 854, 16)
                    side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                    side_5_f = crop_features(side_5_f, im_size)
                    utils.collect_named_outputs(end_points_collection, segnet_stream + '/side-multi5-cr', side_5_f)

                # Build the main volume concat_side (None, 496, 880, 16x4)
                concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], axis=3)

                # Dim reduction - linearly combine concat_side feature maps -> (None, 496, 880, 1)
                net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')

                # Note that the FC layers of the original VGG16 network are not part of the DRIU architecture

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# Set deconvolutional layers to compute bilinear interpolation
def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                raise ValueError('input + output channels need to be the same')
            if h != w:
                raise ValueError('filters need to be square')
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors

# TODO: Move preprocessing to Tensorflow API?
def preprocess_inputs(inputs, segnet_stream='weak'):
    """Preprocess the inputs to adapt them to the network requirements
    Args:
        Image we want to input to the network in (batch_size,W,H,3) or (batch_size,W,H,4) np array
    Returns:
        Image ready to input to the network with means substracted
    """
    assert(len(inputs.shape) == 4)

    if segnet_stream == 'weak':
        new_inputs = np.subtract(inputs.astype(np.float32), np.array((104.00699, 116.66877, 122.67892, 128.), dtype=np.float32))
    else:
        new_inputs = np.subtract(inputs.astype(np.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    # input = tf.subtract(tf.cast(input, tf.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    # input = np.expand_dims(input, axis=0)
    return new_inputs


# TODO: Move preprocessing to Tensorflow API?
def preprocess_labels(labels):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
        Labels (batch_size,W,H) or (batch_size,W,H,1) in numpy array
    Returns:
        Label ready to compute the loss (batch_size,W,H,1)
    """
    assert(len(labels.shape) == 4)

    max_mask = np.max(labels) * 0.5
    labels = np.greater(labels, max_mask).astype(np.float32)
    if len(labels.shape) == 3:
        labels = np.expand_dims(labels, axis=-1)
    # label = tf.cast(np.array(label), tf.float32)
    # max_mask = tf.multiply(tf.reduce_max(label), 0.5)
    # label = tf.cast(tf.greater(label, max_mask), tf.float32)
    # label = tf.expand_dims(tf.expand_dims(label, 0), 3)
    return labels


def load_vgg_imagenet(ckpt_path, segnet_stream='weak'):
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM
    Args:
        Path to the checkpoint, either the 3-channel or 4-channel input version
        segnet_stream: Is this the 3-channel or the 4-channel input version?
    Returns:
        Function that takes a session and initializes the network
    """
    assert(segnet_stream in ['weak','full'])
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    for v in var_to_shape_map:
        if "conv" in v:
            vars_corresp[v] = slim.get_model_variables(v.replace("vgg_16", segnet_stream))[0]
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, vars_corresp)
    return init_fn

def class_balanced_cross_entropy_loss(output, label):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss


def class_balanced_cross_entropy_loss_theoretical(output, label):
    """Theoretical version of the class balanced cross entropy loss to train the network (Produces unstable results)
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    output = tf.nn.sigmoid(output)

    labels_pos = tf.cast(tf.greater(label, 0), tf.float32)
    labels_neg = tf.cast(tf.less(label, 1), tf.float32)

    num_labels_pos = tf.reduce_sum(labels_pos)
    num_labels_neg = tf.reduce_sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg

    loss_pos = tf.reduce_sum(tf.multiply(labels_pos, tf.log(output + 0.00001)))
    loss_neg = tf.reduce_sum(tf.multiply(labels_neg, tf.log(1 - output + 0.00001)))

    final_loss = -num_labels_neg / num_total * loss_pos - num_labels_pos / num_total * loss_neg

    return final_loss


def load_caffe_weights(weights_path):
    """Initialize the network parameters from a .npy caffe weights file
    Args:
    Path to the .npy file containing the value of the network parameters
    Returns:
    Function that takes a session and initializes the network
    """
    osvos_weights = np.load(weights_path).item()
    vars_corresp = dict()
    vars_corresp['osvos/conv1/conv1_1/weights'] = osvos_weights['conv1_1_w']
    vars_corresp['osvos/conv1/conv1_1/biases'] = osvos_weights['conv1_1_b']
    vars_corresp['osvos/conv1/conv1_2/weights'] = osvos_weights['conv1_2_w']
    vars_corresp['osvos/conv1/conv1_2/biases'] = osvos_weights['conv1_2_b']

    vars_corresp['osvos/conv2/conv2_1/weights'] = osvos_weights['conv2_1_w']
    vars_corresp['osvos/conv2/conv2_1/biases'] = osvos_weights['conv2_1_b']
    vars_corresp['osvos/conv2/conv2_2/weights'] = osvos_weights['conv2_2_w']
    vars_corresp['osvos/conv2/conv2_2/biases'] = osvos_weights['conv2_2_b']

    vars_corresp['osvos/conv3/conv3_1/weights'] = osvos_weights['conv3_1_w']
    vars_corresp['osvos/conv3/conv3_1/biases'] = osvos_weights['conv3_1_b']
    vars_corresp['osvos/conv3/conv3_2/weights'] = osvos_weights['conv3_2_w']
    vars_corresp['osvos/conv3/conv3_2/biases'] = osvos_weights['conv3_2_b']
    vars_corresp['osvos/conv3/conv3_3/weights'] = osvos_weights['conv3_3_w']
    vars_corresp['osvos/conv3/conv3_3/biases'] = osvos_weights['conv3_3_b']

    vars_corresp['osvos/conv4/conv4_1/weights'] = osvos_weights['conv4_1_w']
    vars_corresp['osvos/conv4/conv4_1/biases'] = osvos_weights['conv4_1_b']
    vars_corresp['osvos/conv4/conv4_2/weights'] = osvos_weights['conv4_2_w']
    vars_corresp['osvos/conv4/conv4_2/biases'] = osvos_weights['conv4_2_b']
    vars_corresp['osvos/conv4/conv4_3/weights'] = osvos_weights['conv4_3_w']
    vars_corresp['osvos/conv4/conv4_3/biases'] = osvos_weights['conv4_3_b']

    vars_corresp['osvos/conv5/conv5_1/weights'] = osvos_weights['conv5_1_w']
    vars_corresp['osvos/conv5/conv5_1/biases'] = osvos_weights['conv5_1_b']
    vars_corresp['osvos/conv5/conv5_2/weights'] = osvos_weights['conv5_2_w']
    vars_corresp['osvos/conv5/conv5_2/biases'] = osvos_weights['conv5_2_b']
    vars_corresp['osvos/conv5/conv5_3/weights'] = osvos_weights['conv5_3_w']
    vars_corresp['osvos/conv5/conv5_3/biases'] = osvos_weights['conv5_3_b']

    vars_corresp['osvos/conv2_2_16/weights'] = osvos_weights['conv2_2_16_w']
    vars_corresp['osvos/conv2_2_16/biases'] = osvos_weights['conv2_2_16_b']
    vars_corresp['osvos/conv3_3_16/weights'] = osvos_weights['conv3_3_16_w']
    vars_corresp['osvos/conv3_3_16/biases'] = osvos_weights['conv3_3_16_b']
    vars_corresp['osvos/conv4_3_16/weights'] = osvos_weights['conv4_3_16_w']
    vars_corresp['osvos/conv4_3_16/biases'] = osvos_weights['conv4_3_16_b']
    vars_corresp['osvos/conv5_3_16/weights'] = osvos_weights['conv5_3_16_w']
    vars_corresp['osvos/conv5_3_16/biases'] = osvos_weights['conv5_3_16_b']

    vars_corresp['osvos/score-dsn_2/weights'] = osvos_weights['score-dsn_2_w']
    vars_corresp['osvos/score-dsn_2/biases'] = osvos_weights['score-dsn_2_b']
    vars_corresp['osvos/score-dsn_3/weights'] = osvos_weights['score-dsn_3_w']
    vars_corresp['osvos/score-dsn_3/biases'] = osvos_weights['score-dsn_3_b']
    vars_corresp['osvos/score-dsn_4/weights'] = osvos_weights['score-dsn_4_w']
    vars_corresp['osvos/score-dsn_4/biases'] = osvos_weights['score-dsn_4_b']
    vars_corresp['osvos/score-dsn_5/weights'] = osvos_weights['score-dsn_5_w']
    vars_corresp['osvos/score-dsn_5/biases'] = osvos_weights['score-dsn_5_b']

    vars_corresp['osvos/upscore-fuse/weights'] = osvos_weights['new-score-weighting_w']
    vars_corresp['osvos/upscore-fuse/biases'] = osvos_weights['new-score-weighting_b']
    return slim.assign_from_values_fn(vars_corresp)


def parameter_lr(segnet_stream='weak'):
    """Specify the relative learning rate for every parameter. The final learning rate
    in every parameter will be the one defined here multiplied by the global one
    Args:
        segnet_stream: Is this the 3-channel or the 4-channel input version?
    Returns:
        Dictionary with the relative learning rate for every parameter
    """
    assert(segnet_stream in ['weak','full'])
    vars_corresp = dict()
    vars_corresp[segnet_stream + '/conv1/conv1_1/weights'] = 1
    vars_corresp[segnet_stream + '/conv1/conv1_1/biases'] = 2
    vars_corresp[segnet_stream + '/conv1/conv1_2/weights'] = 1
    vars_corresp[segnet_stream + '/conv1/conv1_2/biases'] = 2

    vars_corresp[segnet_stream + '/conv2/conv2_1/weights'] = 1
    vars_corresp[segnet_stream + '/conv2/conv2_1/biases'] = 2
    vars_corresp[segnet_stream + '/conv2/conv2_2/weights'] = 1
    vars_corresp[segnet_stream + '/conv2/conv2_2/biases'] = 2

    vars_corresp[segnet_stream + '/conv3/conv3_1/weights'] = 1
    vars_corresp[segnet_stream + '/conv3/conv3_1/biases'] = 2
    vars_corresp[segnet_stream + '/conv3/conv3_2/weights'] = 1
    vars_corresp[segnet_stream + '/conv3/conv3_2/biases'] = 2
    vars_corresp[segnet_stream + '/conv3/conv3_3/weights'] = 1
    vars_corresp[segnet_stream + '/conv3/conv3_3/biases'] = 2

    vars_corresp[segnet_stream + '/conv4/conv4_1/weights'] = 1
    vars_corresp[segnet_stream + '/conv4/conv4_1/biases'] = 2
    vars_corresp[segnet_stream + '/conv4/conv4_2/weights'] = 1
    vars_corresp[segnet_stream + '/conv4/conv4_2/biases'] = 2
    vars_corresp[segnet_stream + '/conv4/conv4_3/weights'] = 1
    vars_corresp[segnet_stream + '/conv4/conv4_3/biases'] = 2

    vars_corresp[segnet_stream + '/conv5/conv5_1/weights'] = 1
    vars_corresp[segnet_stream + '/conv5/conv5_1/biases'] = 2
    vars_corresp[segnet_stream + '/conv5/conv5_2/weights'] = 1
    vars_corresp[segnet_stream + '/conv5/conv5_2/biases'] = 2
    vars_corresp[segnet_stream + '/conv5/conv5_3/weights'] = 1
    vars_corresp[segnet_stream + '/conv5/conv5_3/biases'] = 2

    vars_corresp[segnet_stream + '/conv2_2_16/weights'] = 1
    vars_corresp[segnet_stream + '/conv2_2_16/biases'] = 2
    vars_corresp[segnet_stream + '/conv3_3_16/weights'] = 1
    vars_corresp[segnet_stream + '/conv3_3_16/biases'] = 2
    vars_corresp[segnet_stream + '/conv4_3_16/weights'] = 1
    vars_corresp[segnet_stream + '/conv4_3_16/biases'] = 2
    vars_corresp[segnet_stream + '/conv5_3_16/weights'] = 1
    vars_corresp[segnet_stream + '/conv5_3_16/biases'] = 2

    vars_corresp[segnet_stream + '/score-dsn_2/weights'] = 0.1
    vars_corresp[segnet_stream + '/score-dsn_2/biases'] = 0.2
    vars_corresp[segnet_stream + '/score-dsn_3/weights'] = 0.1
    vars_corresp[segnet_stream + '/score-dsn_3/biases'] = 0.2
    vars_corresp[segnet_stream + '/score-dsn_4/weights'] = 0.1
    vars_corresp[segnet_stream + '/score-dsn_4/biases'] = 0.2
    vars_corresp[segnet_stream + '/score-dsn_5/weights'] = 0.1
    vars_corresp[segnet_stream + '/score-dsn_5/biases'] = 0.2

    vars_corresp[segnet_stream + '/upscore-fuse/weights'] = 0.01
    vars_corresp[segnet_stream + '/upscore-fuse/biases'] = 0.02
    return vars_corresp


def _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, segnet_stream='weak', iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False, config=None, finetune=1,
           test_image_path=None, ckpt_name='weak'):
    """Train OSVOS
    Args:
    dataset: Reference to a Dataset object instance
    initial_ckpt: Path to the checkpoint to initialize the network (May be parent network or pre-trained Imagenet)
    supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
    learning_rate: Value for the learning rate. It can be a number or an instance to a learning rate object.
    logs_path: Path to store the checkpoints
    max_training_iters: Number of training iterations
    save_step: A checkpoint will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps
    segnet_stream: Binary segmentation network stream; either "appearance stream" or "flow stream" ['weak'|'full']
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size: Size of the training batch
    momentum: Value of the momentum parameter for the Momentum optimizer
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    finetune: Use to select the type of training, 0 for the parent network and 1 for finetunning
    test_image_path: If image path provided, every save_step the result of the network with this image is stored
    ckpt_name: Checkpoint name
    Returns:
    """
    model_name = os.path.join(logs_path, ckpt_name+".ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data:
    # Section "3.3 Binary Segmentation" of the MaskRNN paper and "Figure 2" are inconsistent when it comes to describing
    # the inputs of the two-stream network. In this implementation, we chose the input of the appearance stream
    # 'weak' to be the concatenation of the current frame I<sub>t</sub> and the warped prediction of the previous
    # frame's segmentation mask b<sub>t-1</sub>, denoted as Phi<sub>t-1,t</sub>(b<sub>t-1</sub>). The warping function
    # Phi<sub>t-1,t</sub>(.) transforms the input based on the optical flow fields from frame I<sub>t-1</sub> to
    # frame I<sub>t</sub>.
    # We chose the input of the flow stream 'full' to be the concatenation of the magnitude of the flow field from
    # I<sub>t-1</sub> to I<sub>t</sub> and I<sub>t</sub> to frame I<sub>t+1</sub> and, again, the warped prediction
    # of the previous frame's segmentation mask b<sub>t-1</sub>.
    # The architecture of both streams is identical.
    assert(segnet_stream in ['weak','full'])
    if segnet_stream == 'weak':
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, 4])
    else:
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    input_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])

    # Create the convnet
    with slim.arg_scope(backbone_arg_scope()):
        net, end_points = backbone(input_image, segnet_stream)

    # Print name and shape of each tensor.
    print("Network Layers:")
    for k, v in end_points.items():
        print('   name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Print name and shape of parameter nodes (values not yet initialized)
    print("Network Parameters:")
    for v in slim.get_model_variables():
        print('   name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Initialize weights from pre-trained model
    if finetune == 0:
        init_weights = load_vgg_imagenet(initial_ckpt, segnet_stream)

    # Define loss
    with tf.name_scope('losses'):
        if supervison == 1 or supervison == 2:
            dsn_2_loss = class_balanced_cross_entropy_loss(end_points[segnet_stream + '/score-dsn_2-cr'], input_label)
            tf.summary.scalar('dsn_2_loss', dsn_2_loss)
            dsn_3_loss = class_balanced_cross_entropy_loss(end_points[segnet_stream + '/score-dsn_3-cr'], input_label)
            tf.summary.scalar('dsn_3_loss', dsn_3_loss)
            dsn_4_loss = class_balanced_cross_entropy_loss(end_points[segnet_stream + '/score-dsn_4-cr'], input_label)
            tf.summary.scalar('dsn_4_loss', dsn_4_loss)
            dsn_5_loss = class_balanced_cross_entropy_loss(end_points[segnet_stream + '/score-dsn_5-cr'], input_label)
            tf.summary.scalar('dsn_5_loss', dsn_5_loss)

        main_loss = class_balanced_cross_entropy_loss(net, input_label)
        tf.summary.scalar('main_loss', main_loss)

        if supervison == 1:
            output_loss = dsn_2_loss + dsn_3_loss + dsn_4_loss + dsn_5_loss + main_loss
        elif supervison == 2:
            output_loss = 0.5 * dsn_2_loss + 0.5 * dsn_3_loss + 0.5 * dsn_4_loss + 0.5 * dsn_5_loss + main_loss
        elif supervison == 3:
            output_loss = main_loss
        else:
            sys.exit('Incorrect supervision id, select 1 for supervision of the side outputs, 2 for weak supervision '
                     'of the side outputs and 3 for no supervision of the side outputs')
        total_loss = output_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = {}
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator[ind] = tf.ConditionalAccumulator(grads_and_vars[ind][0].dtype)
        with tf.name_scope('apply_gradient'):
            layer_lr = parameter_lr(segnet_stream)
            grad_accumulator_ops = []
            for var_ind, grad_acc in grad_accumulator.items(): # Phil: was: for var_ind, grad_acc in grad_accumulator.iteritems():
                var_name = str(grads_and_vars[var_ind][1].name).split(':')[0]
                var_grad = grads_and_vars[var_ind][0]
                grad_accumulator_ops.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for var_ind, grad_acc in grad_accumulator.items(): # Phil: was: for var_ind, grad_acc in grad_accumulator.iteritems():
                mean_grads_and_vars.append(
                    (grad_acc.take_grad(iter_mean_grad), grads_and_vars[var_ind][1]))
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)
    # Log training info
    merged_summary_op = tf.summary.merge_all()

    # Log evolution of test image
    if test_image_path is not None:
        probabilities = tf.nn.sigmoid(net)
        img_summary = tf.summary.image("Output probabilities", probabilities, max_outputs=1)
    # Initialize variables
    init = tf.global_variables_initializer()

    # Create objects to record timing and memory of the graph execution
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # Option in the session options=run_options
    # run_metadata = tf.RunMetadata() # Option in the session run_metadata=run_metadata
    # summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
    with tf.Session(config=config) as sess:
        print('Init variable')
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        else:
            # Load pre-trained model
            if finetune == 0:
                print('Initializing from pre-trained imagenet model...')
                init_weights(sess)
            else:
                print('Initializing from specified pre-trained model...')
                # init_weights(sess)
                var_list = []
                for var in tf.global_variables():
                    var_type = var.name.split('/')[-1]
                    if 'weights' in var_type or 'bias' in var_type:
                        var_list.append(var)
                saver_res = tf.train.Saver(var_list=var_list)
                saver_res.restore(sess, initial_ckpt)
            step = 1
        sess.run(interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print('Start training')
        while step < max_training_iters + 1:
            # Average the gradient
            for _ in range(0, iter_mean_grad):
                batch_inputs, batch_labels = dataset.next_batch(batch_size, 'train', segnet_stream)
                inputs = preprocess_inputs(batch_inputs, segnet_stream)
                labels = preprocess_labels(batch_labels)
                run_res = sess.run([total_loss, merged_summary_op] + grad_accumulator_ops,
                                   feed_dict={input_image: inputs, input_label: labels})
                batch_loss = run_res[0]
                summary = run_res[1]

            # Apply the gradients
            sess.run(apply_gradient_op)  # Momentum updates here its statistics

            # Save summary reports
            summary_writer.add_summary(summary, step)

            # Display training status
            if step % display_step == 0:
                print("{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, batch_loss))

            # Save a checkpoint
            if step % save_step == 0:
                if test_image_path is not None:
                    curr_output = sess.run(img_summary, feed_dict={input_image: preprocess_inputs(test_image_path, segnet_stream)})
                    summary_writer.add_summary(curr_output, step)
                save_path = saver.save(sess, model_name, global_step=global_step)
                print("Model saved in file: %s" % save_path)

            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print("Model saved in file: %s" % save_path)

        print('Finished training.')


def train_parent(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                 display_step, global_step, segnet_stream='full', iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False,
                 config=None, test_image_path=None, ckpt_name='full'):
    """Train OSVOS parent network
    Args:
    See _train()
    Returns:
    """
    finetune = 0
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, segnet_stream, iter_mean_grad, batch_size, momentum, resume_training, config, finetune, test_image_path,
           ckpt_name)


def train_finetune(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                   display_step, global_step, segnet_stream='full', iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False,
                   config=None, test_image_path=None, ckpt_name='full'):
    """Finetune OSVOS
    Args:
    See _train()
    Returns:
    """
    finetune = 1
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, segnet_stream, iter_mean_grad, batch_size, momentum, resume_training, config, finetune, test_image_path,
           ckpt_name)


def test(dataset, checkpoint_file, pred_masks_path, img_pred_masks_path, segnet_stream='full', config=None):
    """Test one sequence
    Args:
        dataset: Reference to a Dataset object instance
        checkpoint_path: Path of the checkpoint to use for the evaluation
        segnet_stream: Binary segmentation network stream; either "appearance stream" or "flow stream" ['weak'|'full']
        pred_masks_path: Path to save the individual predicted masks
        img_pred_masks_path: Path to save the composite of the input image overlayed with the predicted masks
        config: Reference to a Configuration object used in the creation of a Session
    Returns:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    assert(segnet_stream in ['weak','full'])
    batch_size = 1
    if segnet_stream == 'weak':
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, 4])
    else:
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create the convnet
    with slim.arg_scope(backbone_arg_scope()):
        net, end_points = backbone(input_image, segnet_stream)
    probabilities = tf.nn.sigmoid(net)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    if not os.path.exists(pred_masks_path):
        os.makedirs(pred_masks_path)
    if not os.path.exists(img_pred_masks_path):
        os.makedirs(img_pred_masks_path)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_file)
        rounds, rounds_left = divmod(dataset.test_size, batch_size)
        if rounds_left:
            rounds += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _round in trange(rounds, ascii=True, ncols=80, desc='Saving predictions as PNGs'):
                samples, output_files = dataset.next_batch(batch_size, 'test', segnet_stream)
                inputs = preprocess_inputs(samples, segnet_stream)
                masks = sess.run(probabilities, feed_dict={input_image: inputs})
                masks = np.where(masks.astype(np.float32) < 162.0/255.0, 0, 255).astype('uint8')
                for mask, output_file in zip(masks, output_files):
                    imsave(os.path.join(pred_masks_path, output_file), mask[:, :, 0])
