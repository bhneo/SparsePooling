
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path
import sys
import shutil
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import scipy.io as scio
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from common.inputs import data_input
from common import utils


SRC = 'Pascal VOC 2010/VOCdevkit/VOC2010'
OUTPUT = '../../../data/Pascal VOC 2010'
CATEGORIES = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']


class BoundingBox(object):
    pass


def GetItem(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1


def GetInt(name, root, index=0):
    # In some XML annotation files, the point values are not integers, but floats.
    # So we add a float function to avoid ValueError.
    return int(float(GetItem(name, root, index)))


def FindNumberBoundingBoxes(root):
    index = 0
    while True:
        if GetInt('xmin', root, index) == -1:
            break
        index += 1
    return index


def ProcessXMLAnnotation(xml_file):
    """Process a single XML file containing a bounding box."""
    # pylint: disable=broad-except
    try:
        tree = ET.parse(xml_file)
    except Exception:
        print('Failed to parse: ' + xml_file, file=sys.stderr)
        return None
    # pylint: enable=broad-except
    root = tree.getroot()

    num_boxes = FindNumberBoundingBoxes(root)
    boxes = []

    for index in range(num_boxes):
        box = BoundingBox()
        # Grab the 'index' annotation.
        box.xmin = GetInt('xmin', root, index)
        box.ymin = GetInt('ymin', root, index)
        box.xmax = GetInt('xmax', root, index)
        box.ymax = GetInt('ymax', root, index)

        box.width = GetInt('width', root)
        box.height = GetInt('height', root)
        box.filename = GetItem('filename', root) + '.JPEG'
        box.label = GetItem('name', root)

        xmin = float(box.xmin) / float(box.width)
        xmax = float(box.xmax) / float(box.width)
        ymin = float(box.ymin) / float(box.height)
        ymax = float(box.ymax) / float(box.height)

        # Some images contain bounding box annotations that
        # extend outside of the supplied image. See, e.g.
        # n03127925/n03127925_147.xml
        # Additionally, for some bounding boxes, the min > max
        # or the box is entirely outside of the image.
        min_x = min(xmin, xmax)
        max_x = max(xmin, xmax)
        box.xmin_scaled = min(max(min_x, 0.0), 1.0)
        box.xmax_scaled = min(max(max_x, 0.0), 1.0)

        min_y = min(ymin, ymax)
        max_y = max(ymin, ymax)
        box.ymin_scaled = min(max(min_y, 0.0), 1.0)
        box.ymax_scaled = min(max(max_y, 0.0), 1.0)

        boxes.append(box)

    return boxes


def image_normalize(image, arch, inverse=False):
    if arch == 'InceptionV3':
        if inverse:
            image = (image + 1) * 127.5 / 255.
            image = tf.clip_by_value(image, 0, 1)
        else:
            image = image / 127.5 - 1
    else:
        mean = [123.68, 116.779, 103.939]
        if inverse:
            image = (image + mean) / 255.
            image = tf.clip_by_value(image, 0, 1)
        else:
            image -= mean
    return image


def find_animals_file():
    datasets = {'train': [], 'valid': []}
    for category in CATEGORIES:
        train_paths = category + '_train.txt'
        with open(os.path.join(SRC, 'ImageSets/Main', train_paths)) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line[-1] == '1':
                    datasets['train'].append((line[0], category))

        valid_paths = category + '_val.txt'
        with open(os.path.join(SRC, 'ImageSets/Main', valid_paths)) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line[-1] == '1':
                    datasets['valid'].append((line[0], category))
    with open(os.path.join(OUTPUT, 'animals_train.txt'), 'w') as f:
        for sample in datasets['train']:
            f.write(sample[0] + ' ' + sample[1])
            f.write('\n')
    with open(os.path.join(OUTPUT, 'animals_valid.txt'), 'w') as f:
        for sample in datasets['valid']:
            f.write(sample[0] + ' ' + sample[1])
            f.write('\n')

    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT, 'animal_train', category))
        os.makedirs(os.path.join(OUTPUT, 'animal_valid', category))
    source = os.path.join(SRC, 'JPEGImages')
    for sample in datasets['train']:
        shutil.copy(os.path.join(source, sample[0] + '.jpg'), os.path.join(OUTPUT, 'animal_train', sample[1]))
    for sample in datasets['valid']:
        shutil.copy(os.path.join(source, sample[0] + '.jpg'), os.path.join(OUTPUT, 'animal_valid', sample[1]))


def test_find_multi_label():
    datasets = {'train': {}, 'valid': {}}
    with open(os.path.join(OUTPUT, 'animals_train.txt')) as f:
        for line in f.readlines():
            line = line.strip().split()
            if line[0] in datasets['train']:
                datasets['train'][line[0]].append(line[1])
            else:
                datasets['train'][line[0]] = [line[1]]
    with open(os.path.join(OUTPUT, 'animals_train_mul.txt'), 'w') as f:
        for sample in datasets['train']:
            label = ''
            for item in datasets['train'][sample]:
                label += item
                label += ' '
            f.write(sample + ' ' + label)
            f.write('\n')

    with open(os.path.join(OUTPUT, 'animals_valid.txt')) as f:
        for line in f.readlines():
            line = line.strip().split()
            if line[0] in datasets['valid']:
                datasets['valid'][line[0]].append(line[1])
            else:
                datasets['valid'][line[0]] = [line[1]]
    with open(os.path.join(OUTPUT, 'animals_valid_mul.txt'), 'w') as f:
        for sample in datasets['valid']:
            label = ''
            for item in datasets['valid'][sample]:
                label += item
                label += ' '
            f.write(sample + ' ' + label)
            f.write('\n')


def test_generate_masks():
    mask_dir = os.path.join(OUTPUT, 'animal_obj_mask')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    write_obj_masks('animals_train_mul.txt', mask_dir)
    write_obj_masks('animals_valid_mul.txt', mask_dir)


def write_obj_masks(source, mask_dir):
    from scipy import misc
    with open(os.path.join(OUTPUT, source)) as f:
        for line in f.readlines():
            line = line.strip().split()
            file = os.path.join(OUTPUT, 'Annotations_Part', line[0] + '.mat')
            labels = line[1:]

            objects = scio.loadmat(file)['anno'][0][0][1][0]
            valid_obj = []
            for obj in objects:
                if obj[0] in labels:
                    valid_obj.append(obj)
            masks = []
            for item in valid_obj:
                masks.append(np.expand_dims(item[2], -1))
            masks = np.concatenate(masks, -1)
            masks = np.sum(masks, -1, keepdims=False)
            masks[masks > 1] = 1
            misc.imsave(os.path.join(mask_dir, line[0] + '.jpg'), masks)


def build_dataset(data_dir='data', batch_size=128, shape=(224, 224, 3), flip=True, crop=True):
    data_path = os.path.join(data_dir, 'Pascal VOC 2010')
    labels = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
    train_path = os.path.join(data_path, 'animal_train_crop')
    valid_path = os.path.join(data_path, 'animal_valid_crop')
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    for i, l in enumerate(labels):
        train_files = os.listdir(os.path.join(train_path, l))
        for file in train_files:
            file = os.path.join(train_path, l, file)
            train_images.append(file)
            train_labels.append(i)
        valid_files = os.listdir(os.path.join(valid_path, l))
        for file in valid_files:
            file = os.path.join(valid_path, l, file)
            valid_images.append(file)
            valid_labels.append(i)

    train_images = tf.constant(train_images)
    train_labels = tf.constant(train_labels)
    valid_images = tf.constant(valid_images)
    valid_labels = tf.constant(valid_labels)

    train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(2816).\
        map(build_parse((shape[0], shape[1]), flip), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels)).\
        map(build_parse((shape[0], shape[1])), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    info = data_input.DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=shape),
                                                           'label': tfds.features.ClassLabel(num_classes=len(labels))}),
                                                          {'train_examples': 2816,
                                                           'test_examples': 2839})
    return train, test, info


def build_dataset2(data_dir='data', batch_size=128, shape=(224, 224, 3), target=None):
    data_path = os.path.join(data_dir, 'Pascal VOC 2010')
    SRC_path = os.path.join(data_dir, SRC)
    train_path = os.path.join(data_path, 'animal_train')
    valid_path = os.path.join(data_path, 'animal_valid')
    train_images = []
    train_boxes = []
    train_labels = []
    valid_images = []
    valid_boxes = []
    valid_labels = []

    for i, l in enumerate(CATEGORIES):
        train_files = os.listdir(os.path.join(train_path, l))
        for file in train_files:
            obj_anno = file.split('.')[0] + '.xml'
            obj_anno = os.path.join(SRC_path, 'Annotations', obj_anno)
            tree = ET.parse(obj_anno)
            file = os.path.join(train_path, l, file)
            train_images.append(file)
            train_labels.append(i)
            # img = Image.open(file).size
            area = 0
            box = []
            for obj in tree.getroot().iter('object'):
                if obj.find('name').text == l:
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    new_area = (xmax-xmin)*(ymax-ymin)
                    if new_area > area:
                        area = new_area
                        box = [ymin, xmin, ymax, xmax]
            train_boxes.append(box)

        valid_files = os.listdir(os.path.join(valid_path, l))
        for file in valid_files:
            obj_anno = file.split('.')[0] + '.xml'
            obj_anno = os.path.join(SRC_path, 'Annotations', obj_anno)
            tree = ET.parse(obj_anno)
            area = 0
            box = []
            for obj in tree.getroot().iter('object'):
                if obj.find('name').text == l:
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    new_area = (xmax - xmin) * (ymax - ymin)
                    if new_area > area:
                        area = new_area
                        box = [ymin, xmin, ymax, xmax]
            valid_boxes.append(box)

            file = os.path.join(valid_path, l, file)
            valid_images.append(file)
            valid_labels.append(i)

    train_images = tf.constant(train_images)
    train_boxes = tf.constant(train_boxes, dtype=tf.float32)
    train_labels = tf.constant(train_labels)
    valid_images = tf.constant(valid_images)
    valid_boxes = tf.constant(valid_boxes, dtype=tf.float32)
    valid_labels = tf.constant(valid_labels)

    train = tf.data.Dataset.from_tensor_slices((train_images, train_boxes, train_labels)).shuffle(1919).\
        map(build_parse2((shape[0], shape[1]), train=True), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test = tf.data.Dataset.from_tensor_slices((valid_images, valid_boxes, valid_labels)).\
        map(build_parse2((shape[0], shape[1])), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    info = data_input.DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=shape),
                                                           'label': tfds.features.ClassLabel(num_classes=len(CATEGORIES))}),
                                                          {'train_examples': 1919,
                                                           'test_examples': 1914})
    if target:
        target = CATEGORIES.index(target)

        def single_parse(image, label):
            label = tf.equal(label, target)
            label = tf.cast(label, tf.int32)
            return image, label
        train = train.map(single_parse)
        test = test.map(single_parse)
    return train, test, info


def get_test_set_with_landmark3(data_dir='data', category=None, batch_size=128, shape=(224, 224, 3), arch='InceptionV3'): # full image
    data_path = os.path.join(data_dir, 'Pascal VOC 2010')
    SRC_path = os.path.join(data_dir, SRC)
    valid_path = os.path.join(data_path, 'animals_valid_mul.txt')
    valid_images = []
    valid_labels = []
    valid_masks = []

    def get_parse(size):
        def parse(path, label, mask):
            image_str = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image_str)
            image = tf.image.resize(image, size=size)
            image = image_normalize(image, arch)
            return image, label, mask
        return parse

    with open(valid_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            file = os.path.join(SRC_path, 'JPEGImages', line[0] + '.jpg')
            labels = line[1:]
            labels_one_hot = np.zeros([6,])
            if category is not None and category not in labels:
                continue
            for label in labels:
                idx = CATEGORIES.index(label)
                labels_one_hot[idx] = 1
            valid_images.append(file)
            valid_labels.append(labels_one_hot)

            part_mask = line[0] + '.mat'
            part_mask = os.path.join(data_path, 'Annotations_Part', part_mask)
            valid_masks.append(part_mask)

    valid_images = tf.constant(valid_images)
    valid_labels = tf.constant(valid_labels)
    valid_landmarks = tf.constant(valid_masks)

    num = len(valid_images)
    test = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels, valid_landmarks)).\
        map(get_parse((shape[0], shape[1])), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    info = data_input.DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=shape),
                                                           'label': tfds.features.ClassLabel(num_classes=6)}),
                               {'test_examples': num})
    return test, info


def build_dataset3(data_dir='data', batch_size=128, shape=(224, 224, 3), target=None, with_mask=False, arch='InceptionV3', multi=False, shuffle_test=False):
    data_path = os.path.join(data_dir, 'Pascal VOC 2010')
    SRC_path = os.path.join(data_dir, SRC, 'JPEGImages')
    mask_path = os.path.join(data_path, 'animal_obj_mask')
    train_path = os.path.join(data_path, 'animals_train_mul.txt')
    valid_path = os.path.join(data_path, 'animals_valid_mul.txt')
    train_images = []
    train_masks = []
    train_labels = []
    valid_images = []
    valid_masks = []
    valid_labels = []

    with open(train_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            if multi and len(line) <= 2:
                continue
            file = os.path.join(SRC_path, line[0] + '.jpg')
            mask_file = os.path.join(mask_path, line[0] + '.jpg')
            train_images.append(file)
            train_masks.append(mask_file)
            labels = line[1:]
            labels_one_hot = np.zeros([6,])
            for label in labels:
                idx = CATEGORIES.index(label)
                labels_one_hot[idx] = 1
            train_labels.append(labels_one_hot)
    with open(valid_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            if multi and len(line) <= 2:
                continue
            file = os.path.join(SRC_path, line[0] + '.jpg')
            mask_file = os.path.join(mask_path, line[0] + '.jpg')
            valid_images.append(file)
            valid_masks.append(mask_file)
            labels = line[1:]
            labels_one_hot = np.zeros([6,])
            for label in labels:
                idx = CATEGORIES.index(label)
                labels_one_hot[idx] = 1
            valid_labels.append(labels_one_hot)

    train_num = len(train_images)
    valid_num = len(valid_images)
    if shuffle_test:
        idx = [i for i in range(valid_num)]
        np.random.shuffle(idx)
        valid_images = np.array(valid_images)[idx]
        valid_labels = np.array(valid_labels)[idx]

    train_images = tf.constant(train_images)
    train_masks = tf.constant(train_masks)
    train_labels = tf.constant(train_labels)
    valid_images = tf.constant(valid_images)
    valid_masks = tf.constant(valid_masks)
    valid_labels = tf.constant(valid_labels)

    train = tf.data.Dataset.from_tensor_slices(((train_images, train_masks), train_labels)).shuffle(train_num).\
        map(build_parse4((shape[0], shape[1]), train=True, with_mask=with_mask, arch=arch), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test = tf.data.Dataset.from_tensor_slices(((valid_images, valid_masks), valid_labels)).\
        map(build_parse4((shape[0], shape[1]), with_mask=with_mask, arch=arch), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    info = data_input.DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=shape),
                                                           'label': tfds.features.ClassLabel(num_classes=6)}),
                                                          {'train_examples': train_num,
                                                           'test_examples': valid_num})
    if target:
        target = CATEGORIES.index(target)

        def single_parse(image, label):
            label = tf.equal(label, target)
            label = tf.cast(label, tf.int32)
            label = tf.reduce_sum(label, -1)
            return image, label
        train = train.map(single_parse)
        test = test.map(single_parse)
    return train, test, info


def multi2single(target, label):
    target = CATEGORIES.index(target)
    if isinstance(label, tf.Tensor):
        label = label.numpy()
    label = label == target
    label = label.astype(int)
    return label


def build_parse2(size, train=False, brightness=False, contrast=False, arch='InceptionV3'):
    def parse(path, bbox, label):
        image_str = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_str)

        if train:
            float_shape = tf.cast(tf.shape(image), tf.float32)
            ymin = bbox[0] / float_shape[0]
            xmin = bbox[1] / float_shape[1]
            ymax = bbox[2] / float_shape[0]
            xmax = bbox[3] / float_shape[1]
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                # tf.image.extract_jpeg_shape(image_str),
                tf.shape(image),
                bounding_boxes=[[[ymin, xmin, ymax, xmax]]],
                min_object_covered=0.1,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.05, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, _ = sample_distorted_bounding_box

            # Reassemble the bounding box in the format the crop op requires.
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)

            # Use the fused decode and crop op here, which is faster than each in series.
            image = tf.image.crop_to_bounding_box(
                    image, offset_y, offset_x, target_height, target_width)
        else:
            bbox = tf.cast(bbox, tf.int32)
            image = tf.image.crop_to_bounding_box(
                image, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

        if train:
            image = tf.image.random_flip_left_right(image)
        # if brightness:
        #     image = tf.image.random_brightness(image, max_delta=63)
        # if contrast:
        #     image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.resize(image, size=size)
        image = image_normalize(image, arch)
        return image, label
    return parse


def build_parse3(size, train=False, brightness=False, contrast=False, with_mask=False, arch='InceptionV3'):
    def parse(path, label):
        image_path, mask_path = path
        label = tf.cast(label, tf.float32)
        image_str = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_str)
        mask_str = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask_str)
        if train:
            if with_mask:
                mask = tf.tile(mask, [1, 1, 3])
                image = tf.concat([image, mask], 0)
            image = tf.image.random_flip_left_right(image)
            if with_mask:
                image, mask = tf.split(image, 2, axis=0)
                mask = tf.split(mask, 3, axis=2)[0]
            if brightness:
                image = tf.image.random_brightness(image, max_delta=63)
            if contrast:
                image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.resize(image, size=size)
        image = image_normalize(image, arch)
        if with_mask:
            mask = tf.image.resize(mask, size=size)
            mask /= 255.
            return (image, mask), label
        else:
            return image, label

    return parse


def build_parse4(size, train=False, with_mask=None, arch='InceptionV3'):
    def parse(path, label):
        image_path, _ = path
        label = tf.cast(label, tf.float32)
        image_str = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_str)
        shape = tf.shape(input=image)
        height, width = shape[0], shape[1]
        if train:
            image = tf.image.random_flip_left_right(image)
            smaller_dim = tf.minimum(height, width)
            image = tf.image.random_crop(image, [smaller_dim-80, smaller_dim-80,3])
            image = tf.image.resize(image, size=size)
        else:
            new_height, new_width = utils.smallest_size_at_least(height, width, size[0])
            image = tf.image.resize(image, size=(new_height+1, new_width+1))
            image = utils.central_crop(image, size[0], size[1])
        image = image_normalize(image, arch)
        return image, label
    return parse


def build_parse(size, flip=False, arch='InceptionV3'):
    def parse(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, size=size)
        if flip:
            image = tf.image.random_flip_left_right(image)
        image = image_normalize(image, arch)
        return image, label
    return parse


def get_test_set_with_landmark(data_dir='data', category=None, batch_size=128, shape=(224, 224, 3), arch='InceptionV3'):
    data_path = os.path.join(data_dir, 'Pascal VOC 2010')
    SRC_path = os.path.join(data_dir, SRC)
    labels = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
    valid_path = os.path.join(data_path, 'animal_valid')
    valid_images = []
    valid_boxes = []
    valid_labels = []
    valid_masks = []

    def get_parse(size):
        def parse(path, bbox, label, mask):
            image_str = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image_str)
            bbox = tf.cast(bbox, tf.int32)
            image = tf.image.crop_to_bounding_box(
                    image, bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            image = tf.image.resize(image, size=size)
            image = image_normalize(image, arch)
            return image, bbox, label, mask

        return parse

    for i, l in enumerate(labels):
        if category is not None and category != l:
            continue
        valid_files = os.listdir(os.path.join(valid_path, l))
        for file in valid_files:
            obj_anno = file.split('.')[0] + '.xml'
            obj_anno = os.path.join(SRC_path, 'Annotations', obj_anno)
            tree = ET.parse(obj_anno)
            boxes = get_boxes(tree, l)
            if len(boxes) != 1:
                continue

            part_mask = file.split('.')[0] + '.mat'
            part_mask = os.path.join(data_path, 'Annotations_Part', part_mask)
            file = os.path.join(valid_path, l, file)

            objects = scio.loadmat(part_mask)['anno'][0][0][1][0]
            valid_obj = []
            for obj in objects:
                if obj[0] == l:
                    valid_obj.append(obj)
            if len(valid_obj) != 1:
                raise Exception('more than 1 obj!')
            if len(valid_obj[0][3]) == 0:
                continue

            valid_boxes += boxes
            valid_images.append(file)
            valid_labels.append(i)
            valid_masks.append(part_mask)

    valid_images = tf.constant(valid_images)
    valid_boxes = tf.constant(valid_boxes, dtype=tf.float32)
    valid_labels = tf.constant(valid_labels)
    valid_landmarks = tf.constant(valid_masks)

    test = tf.data.Dataset.from_tensor_slices((valid_images, valid_boxes, valid_labels, valid_landmarks)).\
        map(get_parse((shape[0], shape[1])), num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    info = data_input.DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=shape),
                                                           'label': tfds.features.ClassLabel(num_classes=len(labels))}),
                               {'test_examples': 1460})
    return test, info


def parse_masks(images, files, labels, boxes):
    filtered_images = []
    filtered_labels = []
    filtered_masks = []
    shape = images.get_shape().as_list()
    h, w = shape[1], shape[2]
    for image, file, label, box in zip(images, files, labels, boxes):
        # file = tf.constant('../../../data\\Pascal VOC 2010\\Annotations_Part\\2009_002002.mat')
        mask = {}
        objects = scio.loadmat(file.numpy())['anno'][0][0][1][0]
        valid_obj = []
        for obj in objects:
            if obj[0] == label:
                valid_obj.append(obj)
        if len(valid_obj) != 1:
            raise Exception('more than 1 obj!')

        for i, item in enumerate(valid_obj[0]):
            if i == 2:
                item = crop_resize_mask(item, box, h, w)
                mask['obj'] = item
            if i == 3:
                parts = {}
                if len(item) == 0:
                    print(item)
                    continue
                for part in item[0]:
                    name = part[0][0]
                    value = part[1]
                    parts[name] = crop_resize_mask(value, box, h, w)
                parts = merge_parts(parts, label, h, w)
                mask['parts'] = parts
        if 'parts' in mask:
            filtered_images.append(image)
            filtered_labels.append(label)
            filtered_masks.append(mask)
    return filtered_images, filtered_labels, filtered_masks


def get_boxes(tree, label):
    boxes = []
    for obj in tree.getroot().iter('object'):
        if obj.find('name').text == label:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            box = [ymin, xmin, ymax, xmax]
            boxes.append(box)
    return boxes


def test_test_landmark():
    batch = 16
    test, info = get_test_set_with_landmark('../../../data', batch_size=batch)
    # test = test.shuffle(1000)
    count1 = 0
    count2 = 0
    for images, boxes, labels, masks in test:
        count1 += images.get_shape().as_list()[0]
        txt_label = [CATEGORIES[l] for l in labels]
        images, labels, masks = parse_masks(images, masks, txt_label, boxes)
        count2 += len(images)
        parts = get_parts(masks)
        viz_image_mask(images, masks)
    print('count1:{}'.format(count1))
    print('count2:{}'.format(count2))


def get_parts(masks):
    result = []
    for mask in masks:
        result.append(mask['parts'])
    return result


def viz_image_mask(images, masks):
    for image, mask in zip(images, masks):
        mask_img = [np.tile(mask['obj'], [1, 1, 3])]
        for part in mask['parts']:
            part_img = mask['parts'][part]
            part_img = np.tile(part_img, [1, 1, 3])
            mask_img.append(part_img)
        show = np.concatenate([image] + mask_img, 1)
        plt.imshow(show)
        plt.show()


def crop_resize_mask(mask, box, h, w):
    mask = mask[box[0]:box[2], box[1]:box[3], np.newaxis]
    mask = tf.image.resize(mask, [h, w]).numpy()
    mask[mask > 0] = 1
    return mask


def merge_parts(parts, label, h, w):
    results = {}
    if label == 'bird':
        head = []
        torso = []
        leg = []
        tail = []
        results = {'head': head, 'torso': torso, 'leg': leg, 'tail': tail}
        for part in parts:
            if part in ['head', 'leye', 'reye', 'beak']:
                head.append(parts[part])
            if part in ['torso', 'neck', 'lwing', 'rwing']:
                torso.append(parts[part])
            if part in ['lleg', 'rleg', 'lfoot', 'rfoot']:
                leg.append(parts[part])
            if part in ['tail']:
                tail.append(parts[part])
    if label in ['cat', 'dog', 'cow', 'sheep', 'horse']:
        head = []
        torso = []
        bleg = []
        fleg = []
        tail = []
        results = {'head': head, 'torso': torso, 'bleg': bleg, 'fleg': fleg}
        if label in ['cat', 'dog']:
            results['tail'] = tail
        for part in parts:
            if part in ['head', 'leye', 'reye', 'lear', 'rear', 'nose', 'muzzle', 'rhorn', 'lhorn']:
                head.append(parts[part])
            if part in ['torso', 'neck']:
                torso.append(parts[part])
            if part in ['lbleg', 'rbleg', 'lbpa', 'rbpa', 'lblleg', 'lbuleg', 'rblleg', 'rbuleg', 'rbho', 'lbho']:
                bleg.append(parts[part])
            if part in ['lfleg', 'rfleg', 'lfpa', 'rfpa', 'lflleg', 'lfuleg', 'rflleg', 'rfuleg', 'rfho', 'lfho']:
                fleg.append(parts[part])
            if part in ['tail']:
                tail.append(parts[part])

    final = {}
    for merged in results:
        if len(results[merged]) > 1:
            summed = np.sum(results[merged], 0)
            summed[summed > 0] = 1
            final[merged] = summed
        elif len(results[merged]) == 1:
            summed = results[merged][0]
            summed[summed > 0] = 1
            final[merged] = summed
        elif len(results[merged]) == 0:
            final[merged] = np.zeros(shape=(h, w, 1))
    return final


def test_find_shapes():
    from scipy import misc
    data_dir = '../../../data'
    data_path = os.path.join(data_dir, 'Pascal VOC 2010')
    SRC_path = os.path.join(data_dir, SRC, 'JPEGImages')
    train_path = os.path.join(data_path, 'animals_train_mul.txt')
    valid_path = os.path.join(data_path, 'animals_valid_mul.txt')
    hs = []
    ws = []
    with open(train_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            file = os.path.join(SRC_path, line[0] + '.jpg')
            image = misc.imread(file)
            h, w = image.shape[0], image.shape[1]
            hs.append(h)
            ws.append(w)
    print('h min:{},h max:{}'.format(min(hs), max(hs)))
    print('w min:{},w max:{}'.format(min(ws), max(ws)))


def test_read():
    train, test, info = build_dataset3('../../../data', multi=True)
    count = 0
    for image, label in train:
        count += image.shape[0]
    print('train num', count)
    count = 0
    for image, label in test:
        count += image.shape[0]
    print('test num', count)


def test_view_data():
    train, test, info = build_dataset3('../../../data', with_mask=False, multi=True)
    for image, label in train:
        # image, mask = image
        # h,w =image
        image = (image + 1)*127.5/255
        # mask = np.tile(mask, [1, 1, 1, 3])
        # image = np.concatenate([image, mask], 2)
        out_image(image, label)
        break

    for image, label in test:
        # image, mask = image
        image = (image + 1)*127.5/255
        out_image(image, label)
        break


def out_image(images, labels, preds=None, photos=16):
    fig = plt.figure()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, bottom=0.05, right=0.95, left=0.05)
    for i in range(photos):
        plt.subplot(photos/2, 2, i+1)
        plt.axis('off')
        if preds is None:
            title = str(labels[i])
        else:
            title = str(labels[i]) + '_' + str(preds[i])
        plt.title(title)
        image = images[i, :, :, :]
        if image.shape[-1] == 1:
            image = np.squeeze(image, -1)
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

