"""
dataset.py

Dataset utility functions and classes.

Following the instructions provided in Section "6. Instance Segmentation Results" of the "Simple Does It" paper, we use
the Berkeley-augmented Pascal VOC segmentation dataset that provides per-instance segmentation masks for VOC2012 data.
The Berkley augmented dataset can be downloaded from here:
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/tf_records.py
    https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/pascal_voc.py
    https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/recipes/pascal_voc/convert_pascal_voc_to_tfrecords.ipynb
    Copyright (c) 2017 Daniil Pakhomov / Written by Daniil Pakhomov
    Licensed under the MIT License

More to look at later to add support for TFRecords:
  https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/base.py
  https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/loader.py
  https://github.com/kwotsin/create_tfrecords
  https://kwotsin.github.io/tech/2017/01/29/tfrecords.html
  http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/
  - http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    How to write into and read from a tfrecords file in TensorFlow
    Writeen by Hadi Kazemi
  - https://github.com/ferreirafabio/video2tfrecords/blob/master/video2tfrecords.py
    Copyright (c) 2017 Fábio Ferreira / Written Fábio Ferreira
    Licensed under the MIT License
"""

# TODO Add support for TFRecords

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import glob
import warnings
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from bboxes import extract_bbox
from segment import rect_mask, grabcut
from visualize import draw_masks

if sys.platform.startswith("win"):
    _BK_VOC_DATASET = "E:/datasets/bk-voc/benchmark_RELEASE/dataset"
else:
    _BK_VOC_DATASET = '/media/EDrive/datasets/bk-voc/benchmark_RELEASE/dataset'

_DBG_TRAIN_SET = -1

_DEFAULT_BKVOC_OPTIONS = {
    'in_memory': False,
    'data_aug': False,
    'use_cache': False,
    'use_grabcut_labels': True}

class BKVOCDataset(object):
    """Berkeley-augmented Pascal VOC 2012 segmentation dataset.
    """

    def __init__(self, phase='train', dataset_root=_BK_VOC_DATASET, options=_DEFAULT_BKVOC_OPTIONS):
        """Initialize the Dataset object
        Args:
            phase: Possible options: 'train' or 'test'
            dataset_root: Path to the root of the dataset
            options: see below
        Options:
            in_memory: True loads all the training images upfront, False loads images in small batches
            data_aug: True adds augmented data to training set
            use_cache: True stores training files and augmented versions in npy file
            use_grabcut_labels: True computes magnitudes of forward and backward flows
        """
        # Only options supported in this initial implementation
        assert (options == _DEFAULT_BKVOC_OPTIONS)

        # Save file and folder name
        self._dataset_root = dataset_root
        self._phase = phase
        self._options = options

        # Set paths and file names
        self._img_folder = self._dataset_root + '/img'
        self._mats_folder = self._dataset_root + '/inst'
        self._masks_folder = self._dataset_root + '/inst_masks'
        self._grabcuts_folder = self._dataset_root + '/inst_grabcuts'
        self.pred_masks_path = self._dataset_root + '/predicted_inst_masks'
        self.img_pred_masks_path = self._dataset_root + '/img_with_predicted_inst_masks'
        self._train_IDs_file = self._dataset_root + '/train.txt'
        self._test_IDs_file = self._dataset_root + '/val.txt'
        self._img_mask_pairs_file = self._dataset_root + '/img_mask_pairs.txt'
        self._train_img_mask_pairs_file = self._dataset_root + '/train_img_mask_pairs.txt'
        self._test_img_mask_pairs_file = self._dataset_root + '/val_img_mask_pairs.txt'

        # Load ID files
        if not self._load_img_mask_pairs_file(self._img_mask_pairs_file):
            self.prepare()

        # Init batch parameters
        if self._phase == 'train':
            self._load_img_mask_pairs_file(self._train_img_mask_pairs_file)
            self._grabcut_files = [self._grabcuts_folder + '/' + os.path.basename(img_mask_pair[1]) for img_mask_pair in self._img_mask_pairs]
            self._train_ptr = 0
            self.train_size = len(self._img_mask_pairs) if _DBG_TRAIN_SET == -1 else _DBG_TRAIN_SET
            self._train_idx = np.arange(self.train_size)
            np.random.seed(1)
            np.random.shuffle(self._train_idx)
        else:
            self._options['use_grabcut_labels'] = False
            self._load_img_mask_pairs_file(self._test_img_mask_pairs_file)
            self._test_ptr = 0
            self.test_size = len(self._img_mask_pairs)

    ###
    ### Input Samples and Labels Prep
    ###
    def prepare(self):
        """Do all the preprocessing needed before training/val/test samples can be generated.
        """
        # Convert instance masks stored in .mat files to .png files and compute their bboxes
        self._mat_masks_to_png()
    
        # Generate grabcuts, if they don't exist yet
        self._bboxes_to_grabcuts()
    
        # Generate train and test image/mask pair files, if they don't exist yet
        self._split_img_mask_pairs()

    def _save_img_mask_pairs_file(self):
        """Create the file that matches masks with their image file (and has bbox info)
        """
        assert (len(self._mask_bboxes) == len(self._img_mask_pairs))
        with open(self._img_mask_pairs_file, 'w') as img_mask_pairs_file:
            for img_mask_pair, mask_bbox in zip(self._img_mask_pairs, self._mask_bboxes):
                img_path = os.path.basename(img_mask_pair[0])
                mask_path = os.path.basename(img_mask_pair[1])
                line = '{}###{}###{}###{}###{}###{}\n'.format(img_path, mask_path, mask_bbox[0],
                                                              mask_bbox[1], mask_bbox[2], mask_bbox[3])
                img_mask_pairs_file.write(line)
    
    
    def _split_img_mask_pairs(self):
        """Create the training and test portions of the image mask pairs
        """
        if os.path.exists(self._train_img_mask_pairs_file) and os.path.exists(self._test_img_mask_pairs_file):
            return False

        with open(self._train_IDs_file, 'r') as f:
            train_IDs = f.readlines()

        with open(self._test_IDs_file, 'r') as f:
            test_IDs = f.readlines()

        # Load complete list of entries and separate training and test entries
        assert(os.path.exists(self._img_mask_pairs_file))
        with open(self._img_mask_pairs_file, 'r') as img_mask_pairs_file:
            lines = img_mask_pairs_file.readlines()
            train_lines = []
            test_lines = []
            for line in lines:
                splits = line.split('###')
                file_ID = '{}\n'.format(str(splits[0])[-15:-4])
                if file_ID in train_IDs:
                    train_lines.append(line)
                elif file_ID in test_IDs:
                    test_lines.append(line)
                else:
                    raise ValueError('Error in processing train/val text files.')

        # Save result
        with open(self._train_img_mask_pairs_file, 'w') as f:
            for line in train_lines:
                f.write(line)
        with open(self._test_img_mask_pairs_file, 'w') as f:
            for line in test_lines:
                f.write(line)
        return True


    def _load_img_mask_pairs_file(self, img_mask_pairs_path):
        """Load the file that matches masks with their image file (and has bbox info)
        Args:
            img_mask_pairs_path: path to file
        Returns:
          True if file was correctly loaded, False otherwise
        """
        if os.path.exists(img_mask_pairs_path):
            with open(img_mask_pairs_path, 'r') as img_mask_pairs_file:
                lines = img_mask_pairs_file.readlines()
                self._img_mask_pairs = []
                self._mask_bboxes = []
                for line in lines:
                    splits = line.split('###')
                    img_path = self._img_folder + '/' + str(splits[0])
                    mask_path = self._masks_folder + '/' + str(splits[1])
                    self._img_mask_pairs.append((img_path, mask_path))
                    self._mask_bboxes.append((int(splits[2]), int(splits[3]), int(splits[4]), int(splits[5])))
                return True
        return False
    
    
    def _mat_masks_to_png(self):
        """Converts instance masks stored in .mat files to .png files.
        PNG files are created in the same folder as where the .mat files are.
        If the name of this folder ends with "cls", class masks are created.
        If the name of this folder ends with "inst", instance masks are created.
    
        Returns:
          True if files were created, False if the masks folder already contains PNG files
        """
        mat_files = glob.glob(self._mats_folder + '/*.mat')
    
        # Build the list of image files for which we have mat masks
        key = os.path.basename(os.path.normpath(self._mats_folder))
        if key == 'cls':
            key = 'GTcls'
        elif key == 'inst':
            key = 'GTinst'
        else:
            raise ValueError('ERR: Expected mask folder path to end with "/inst" or "/cls"')
    
        # Create output folder, if necessary
        img_files = [self._img_folder + '/' + os.path.basename(file).replace('.mat', '.jpg') for file in mat_files]
        if not os.path.exists(self._masks_folder):
            os.makedirs(self._masks_folder)
    
        # Generate image mask pairs and compute their bboxes
        self._img_mask_pairs = []
        self._mask_bboxes = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mat_file, img_file in tqdm(zip(mat_files, img_files), total=len(mat_files), ascii=True, ncols=80,
                                           desc='MAT to PNG masks'):
                mat = loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
                masks = mat[key].Segmentation
                mask_file_basename = os.path.basename(mat_file)
                for instance in np.unique(masks)[1:]:
                    mask_file = self._masks_folder + '/' + mask_file_basename[:-4] + '_' + str(int(instance)) + '.png'
                    self._img_mask_pairs.append((img_file, mask_file))
                    # Build mask for object instance
                    mask = img_as_ubyte(masks == instance)
                    # Compute the mask's bbox
                    self._mask_bboxes.append(extract_bbox(mask))
                    # Save the mask in PNG format to mask folder
                    imsave(mask_file, mask)
    
        # Save the results to disk
        self._save_img_mask_pairs_file()
    
        return True
    
    
    def _bboxes_to_grabcuts(self):
        """Generate segmentation masks from images and bounding boxes using Grabcut.
        """
        mask_files = glob.glob(self._masks_folder + '/*.png')
        if os.path.exists(self._grabcuts_folder):
            self._grabcut_files = glob.glob(self._grabcuts_folder + '/*.png')
            if _DBG_TRAIN_SET == -1:
                if self._grabcut_files and len(self._grabcut_files) == len(mask_files):
                    return False
            else:
                if self._grabcut_files and len(self._grabcut_files) >= _DBG_TRAIN_SET:
                    return False

        # Create output folder, if necessary
        grabcut_files = [self._grabcuts_folder + '/' + os.path.basename(img_mask_pair[1]) for img_mask_pair in
                         self._img_mask_pairs]
        if not os.path.exists(self._grabcuts_folder):
            os.makedirs(self._grabcuts_folder)
    
        # Run Grabcut on input data
        self._grabcut_files = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for img_mask_pair, mask_bbox, grabcut_file in tqdm(zip(self._img_mask_pairs, self._mask_bboxes, grabcut_files),
                                                               total=len(self._img_mask_pairs), ascii=True, ncols=80,
                                                               desc='Grabcuts'):
                # Continue generating grabcuts from where you last stopped after OpenCV crash
                if not os.path.exists(grabcut_file):
                    # Use Grabcut to create a segmentation within the bbox
                    mask = grabcut(img_mask_pair[0], mask_bbox)
                    # Save the mask in PNG format to the grabcuts folder
                    imsave(grabcut_file, mask)
                self._grabcut_files.append(grabcut_file)
    
        return True

    ###
    ### Batch Management
    ###
    def _load_sample(self, input_rgb_path, input_bbox, label_path=None):
        """Load a propertly formatted sample (input sample + associated label)
        In training mode, there is a label; in testimg mode, there isn't.
        Args:
            input_rgb_path: Path to RGB image
            input_bbox: Bounding box to convert to a binary mask
            label_path: Path to grabcut label, if any label
        Returns in training:
            input sample: RGB+bbox binary mask concatenated in format [W, H, 4]
            label: Grabcut segmentation in format [W, H, 1], if any label
        """
        input_rgb = imread(input_rgb_path)
        input_shape = input_rgb.shape
        input_bin_mask = rect_mask((input_shape[0], input_shape[1], 1), input_bbox)
        assert (len(input_bin_mask.shape) == 3 and input_bin_mask.shape[2] == 1)
        input = np.concatenate((input_rgb, input_bin_mask), axis=-1)
        assert (len(input.shape) == 3 and input.shape[2] == 4)
        if label_path:
            label = imread(label_path)
            label = np.expand_dims(label, axis=-1)
            assert (len(label.shape) == 3 and label.shape[2] == 1)
        else:
            label = None
        return input, label

    def next_batch(self, batch_size, phase='train', segnet_stream='weak'):
        """Get next batch of image (path) and masks
        Args:
            batch_size: Size of the batch
            phase: Possible options:'train' or 'test'
            segnet_stream: Binary segmentation net stream ['weak'|'full']
        Returns in training:
            inputs: Batch of 4-channel inputs (RGB+bbox binary mask) in format [batch_size, W, H, 4]
            labels: Batch of grabcut segmentations in format [batch_size, W, H, 1]
        Returns in testing:
            inputs: Batch of 4-channel inputs (RGB+bbox binary mask) in format [batch_size, W, H, 4]
            output_file: List of output file names that match the bbox file names
        """
        assert (self._options['in_memory'] is False)  # Only option supported at this point
        assert (segnet_stream == 'weak')  # Only option supported at this point
        if phase == 'train':
            inputs, labels = [], []
            if self._train_ptr + batch_size < self.train_size:
                idx = np.array(self._train_idx[self._train_ptr:self._train_ptr + batch_size])
                for l in idx:
                    input, label = self._load_sample(self._img_mask_pairs[l][0], self._mask_bboxes[l],
                                   self._grabcut_files[l])
                    inputs.append(input)
                    labels.append(label)
                self._train_ptr += batch_size
            else:
                old_idx = np.array(self._train_idx[self._train_ptr:])
                np.random.shuffle(self._train_idx)
                new_ptr = (self._train_ptr + batch_size) % self.train_size
                idx = np.array(self._train_idx[:new_ptr])
                inputs_1, labels_1, inputs_2, labels_2 = [], [], [], []
                for l in old_idx:
                    input, label = self._load_sample(self._img_mask_pairs[l][0], self._mask_bboxes[l],
                                   self._grabcut_files[l])
                    inputs_1.append(input)
                    labels_1.append(label)
                for l in idx:
                    input, label = self._load_sample(self._img_mask_pairs[l][0], self._mask_bboxes[l],
                                   self._grabcut_files[l])
                    inputs_2.append(input)
                    labels_2.append(label)
                inputs = inputs_1 + inputs_2
                labels = labels_1 + labels_2
                self._train_ptr = new_ptr
            return np.asarray(inputs), np.asarray(labels)
        elif phase == 'test':
            inputs, output_files = [], []
            if self._test_ptr + batch_size < self.test_size:
                for l in range(self._test_ptr, self._test_ptr + batch_size):
                    input, _ = self._load_sample(self._img_mask_pairs[l][0], self._mask_bboxes[l])
                    output_file = os.path.basename(self._img_mask_pairs[l][1])
                    inputs.append(input)
                    output_files.append(output_file)
                self._test_ptr += batch_size
            else:
                new_ptr = (self._test_ptr + batch_size) % self.test_size
                inputs_1, output_files_1, inputs_2, output_files_2 = [], [], [], []
                for l in range(self._test_ptr, self.test_size):
                    input, _ = self._load_sample(self._img_mask_pairs[l][0], self._mask_bboxes[l])
                    output_file = os.path.basename(self._img_mask_pairs[l][1])
                    inputs_1.append(input)
                    output_files_1.append(output_file)
                for l in range(0, new_ptr):
                    input, _ = self._load_sample(self._img_mask_pairs[l][0], self._mask_bboxes[l])
                    output_file = os.path.basename(self._img_mask_pairs[l][1])
                    inputs_2.append(input)
                    output_files_2.append(output_file)
                inputs = inputs_1 + inputs_2
                output_files = output_files_1 + output_files_2
                self._test_ptr = new_ptr
            return np.asarray(inputs), output_files
        else:
            return None, None

    ###
    ### Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nConfiguration:")
        for k, v in self._options.items():
            print("  {:20} {}".format(k, v))
        print("  {:20} {}".format('phase', self._phase))
        print("  {:20} {}".format('samples', len(self._img_mask_pairs)))

    ###
    ### TODO TFRecords helpers
    ### See:
    ### https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/base.py
    ### https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/loader.py
    ### https://github.com/kwotsin/create_tfrecords
    ### https://kwotsin.github.io/tech/2017/01/29/tfrecords.html
    ### http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/
    ### E:\repos\models-master\research\inception\inception\data\build_imagenet_data.py
    ### E:\repos\models-master\research\object_detection\dataset_tools\create_kitti_tf_record.py
    ###
    def _load_from_tfrecords(self):
        # TODO _load_from_tfrecords
        pass

    def _write_to_tfrecords(self):
        # TODO _write_to_tfrecords
        pass

    def combine_images_with_predicted_masks(self):
        """Build list of individual test immages with predicted masks overlayed."""
        # Overlay masks on top of images
        prev_image, bboxes, masks = None, [], []
        with tqdm(total=len(self._mask_bboxes), desc="Combining JPGs with predictions", ascii=True, ncols=80) as pbar:
            for img_mask_pair, bbox in zip(self._img_mask_pairs, self._mask_bboxes):
                pbar.update(1)
                if img_mask_pair[0] == prev_image:
                    # Accumulate predicted masks and bbox instances belonging to the same image
                    bboxes.append(bbox)
                    masks.append(self.pred_masks_path + '/' + os.path.basename(img_mask_pair[1]))
                else:
                    if prev_image:
                        # Combine image, masks and bboxes in a single image and save the result to disk
                        image = imread(prev_image)
                        masks = np.asarray([imread(mask) for mask in masks])
                        masks = np.expand_dims(masks, axis=-1)
                        draw_masks(image, np.asarray(bboxes), np.asarray(masks))
                        imsave(self.img_pred_masks_path + '/' + os.path.basename(prev_image), image)
                    prev_image = img_mask_pair[0]
                    bboxes = [bbox]
                    masks = [self.pred_masks_path + '/' + os.path.basename(img_mask_pair[1])]

# def test():
#     dataset = BKVOCDataset()
#     dataset.print_config()
#     # WARNING: THE NEXT LINE WILL FORCE REGENERATION OF INTERMEDIARY FILES
#     # dataset.prepare()
#
# if __name__ == '__main__':
#     test()
