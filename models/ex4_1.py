import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from common.inputs import data_input
from common import layers, losses, utils, train

import config


kernel_initializer = keras.initializers.he_normal()
BASE_NAME = 'ex4_1'


def build_model_name(params):
    model_name = BASE_NAME
    model_name += '_{}'.format(params.routing.type)
    model_name += '_iter{}'.format(params.routing.iter_num)
    model_name += '_temper{}'.format(params.routing.temper)
    model_name += '_atoms{}'.format(params.caps.atoms)

    model_name += '_trial{}'.format(str(params.training.idx))

    if params.dataset.flip:
        model_name += '_flip'
    if params.dataset.crop:
        model_name += '_crop'
    return model_name


def build_model(shape, num_out, params):
    inputs = keras.Input(shape=shape)
    model_name = build_model_name(params)
    probs, tensor_log = build(inputs, num_out,
                              params.routing.type,
                              params.routing.iter_num,
                              params.routing.temper,
                              params.caps.atoms)
    model = keras.Model(inputs=inputs, outputs=probs, name=model_name)
    log_model = keras.Model(inputs=inputs, outputs=tensor_log.get_outputs(), name=model_name + '_log')
    tensor_log.set_model(log_model)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=losses.MarginLoss(upper_margin=0.9, bottom_margin=0.1, down_weight=0.5),
                  metrics=[])
    model.summary()

    model.callbacks = []
    return model, tensor_log


def build(inputs, num_out, routing, iter_num, temper, atoms):
    log = utils.TensorLog()
    conv1 = keras.layers.Conv2D(filters=64,
                                kernel_size=9,
                                strides=1,
                                padding='valid',
                                activation='relu')(inputs)
    if routing == 'DR':
        child_pose, child_prob = layers.PrimaryCapsule(kernel_size=9,
                                                       strides=2,
                                                       padding='valid',
                                                       groups=1,
                                                       use_bias=True,
                                                       atoms=64,
                                                       activation='squash',
                                                       kernel_initializer=kernel_initializer)(conv1)
        log.add_hist('child_activation', child_prob)
        log.add_hist('child_pose', child_pose)
        transformed_caps = layers.CapsuleTransformDense(num_out=num_out,
                                                        out_atom=atoms,
                                                        share_weights=False,
                                                        initializer=keras.initializers.glorot_normal())(child_pose)
        parent_poses, parent_probs, cs = layers.DynamicRouting(num_routing=iter_num,
                                                               softmax_in=False,
                                                               temper=temper,
                                                               activation='squash',
                                                               pooling=False,
                                                               log=log)((transformed_caps, child_prob))
        log.add_hist('parent_activation', parent_probs[-1])
        log.add_hist('parent_poses', parent_poses[-1])
        output = parent_probs[-1]
    elif routing == 'EM':
        child_pose, child_prob = layers.PrimaryCapsule(kernel_size=9,
                                                       strides=2,
                                                       padding='valid',
                                                       groups=1,
                                                       use_bias=True,
                                                       atoms=64,
                                                       activation='sigmoid',
                                                       kernel_initializer=kernel_initializer)(conv1)
        log.add_hist('child_activation', child_prob)
        log.add_hist('child_pose', child_pose)
        transformed_caps = layers.CapsuleTransformDense(num_out=num_out,
                                                        out_atom=atoms,
                                                        share_weights=False,
                                                        initializer=keras.initializers.glorot_normal())(child_pose)
        parent_poses, parent_probs, cs = layers.EMRouting(num_routing=iter_num,
                                                          temper=temper,
                                                          log=log)((transformed_caps, child_prob))
        log.add_hist('parent_activation', parent_probs[-1])
        log.add_hist('parent_poses', parent_poses[-1])
        output = parent_probs[-1]
    return output, log


def get_norm_fn(dataset):
    channel = 1
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'svhn_cropped':
        channel = 3

    def norm(image):
        if channel == 3:
            image = tf.image.per_image_standardization(image)
        return image
    return norm


def build_parse(dataset, flip=False, crop=False, is_train=False, with_norm=True):
    if dataset not in ['cifar10', 'cifar100', 'mnist', 'kmnist', 'emnist', 'fashion_mnist', 'svhn_cropped']:
        raise Exception('{} not support!'.format(dataset))
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'svhn_cropped':
        height, width, channel = 32, 32, 3
    if dataset == 'mnist' or dataset == 'kmnist' or dataset == 'fashion_mnist' or dataset == 'emnist':
        height, width, channel = 28, 28, 1

    def parse(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255.)
        if with_norm:
            image = get_norm_fn(dataset)(image)
        if is_train:
            if flip:
                image = tf.image.random_flip_left_right(image)
            if crop:
                image = tf.image.resize_with_crop_or_pad(image, height+8, width+8)
                image = tf.image.random_crop(image, [height, width, channel])
        return image, label
    return parse


def main():
    args, params = config.parse_args()
    if params.task == 'train':
        train_set, test_set, info = data_input.build_dataset(params.dataset.name,
                                                             parser_train=build_parse(params.dataset.name,
                                                                                      flip=params.dataset.flip,
                                                                                      crop=params.dataset.crop,
                                                                                      is_train=True),
                                                             parser_test=build_parse(params.dataset.name,
                                                                                     is_train=False),
                                                             batch_size=params.training.batch_size)
        model, tensor_log = build_model(shape=info.features['image'].shape,
                                        num_out=info.features['label'].num_classes,
                                        params=params)

        trainer = train.Trainer(model, params, info, tensor_log)
        if args.train:
            trainer.fit(train_set, test_set)
        else:
            trainer.evaluate(test_set)
    elif params.task == 'speed':
        get_speed(os.getcwd())


def load_ckpt(model, model_dir):
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=losses.MarginLoss(upper_margin=0.9, bottom_margin=0.1, down_weight=0.5),
                  metrics=[])
    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))


def print_results(data, root='../../', log='log', routing='DR', iter_num=1, temper=1.0):
    model_dir, data_shape, num_out, flip, crop = get_model_dir(data, root + log, routing, iter_num, temper, 16,
                                                               idx=1)
    model = load_model(data_shape, model_dir, num_out, routing, iter_num, temper, 16)
    train_set, test_set, info = data_input.build_dataset(data,
                                                         path=root + 'data',
                                                         parser_train=build_parse(data,
                                                                                  flip=False,
                                                                                  crop=False,
                                                                                  is_train=True),
                                                         parser_test=build_parse(data,
                                                                                 is_train=False),
                                                         batch_size=128)
    metric = keras.metrics.SparseCategoricalAccuracy()
    utils.calculate_time(test_set, model, metric)


def get_speed(root):
    dataset = 'cifar10'
    routing = 'EM'
    iter_nums = [1,2,3,4]
    tempers=[1.0]
    for iter_num in iter_nums:
        for temper in tempers:
            print_results(dataset, root=root, routing=routing, iter_num=iter_num, temper=temper)


def get_input_set(dataset):
    if dataset == 'fashion_mnist' or dataset == 'kmnist':
        data_shape = (28, 28, 1)
        num_out = 10
        flip = False
        crop = True
    elif dataset == 'cifar10':
        data_shape = (32, 32, 3)
        num_out = 10
        flip = True
        crop = True
    elif dataset == 'svhn_cropped':
        data_shape = (32, 32, 3)
        num_out = 10
        flip = False
        crop = True
    return data_shape, num_out, flip, crop


def get_model_dir(dataset, log='log', routing='DR', iter_num=10, temper=1.0, atoms=16, idx=1):
    data_shape, num_out, flip, crop = get_input_set(dataset)
    model_dir = '{}/{}/{}_{}_iter{}_temper{}_atoms{}_trial{}'.format(log, dataset, BASE_NAME, routing,
                                                                     iter_num, temper, atoms, idx)
    if flip:
        model_dir += '_flip'
    if crop:
        model_dir += '_crop'

    if not os.path.exists(model_dir):
        raise Exception('model not exist:{}'.format(model_dir))
    return model_dir, data_shape, num_out, flip, crop


def load_model(data_shape, model_dir, num_out, routing, iter_num=10, temper=1.0, atoms=16, input_norm=None):
    inputs = keras.Input(data_shape)
    probs, log = build(inputs if input_norm is None else layers.InputNorm(input_norm)(inputs),
                       num_out,
                       routing=routing,
                       iter_num=iter_num,
                       temper=temper,
                       atoms=atoms)
    model = keras.Model(inputs=inputs, outputs=probs, name='x')
    load_ckpt(model, model_dir)
    return model


if __name__ == "__main__":
    main()
