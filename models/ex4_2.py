import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from common.inputs import data_input
from common import layers, utils, train, res_blocks, losses, attacks
from common.ops.routing import activated_entropy, coupling_entropy

import config


WEIGHT_DECAY = 1e-4
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_DECAY = 0.99

kernel_regularizer = keras.regularizers.l2(WEIGHT_DECAY)
kernel_initializer = keras.initializers.he_normal()
BASE_NAME = 'ex4_2'


def build_model_name(params):
    model_name = BASE_NAME
    model_name += '_b{}'.format(params.model.resblock)
    model_name += '_{}'.format(params.routing.type)
    if params.routing.type == 'DR' or params.routing.type == 'EM':
        model_name += '_iter{}'.format(params.routing.iter_num)
        model_name += '_temper{}'.format(params.routing.temper)
        model_name += '_atoms{}'.format(params.caps.atoms)

    model_name += '_trial{}'.format(str(params.training.idx))

    if params.dataset.flip:
        model_name += '_flip'
    if params.dataset.crop:
        model_name += '_crop'
    return model_name


def get_loss_opt(type):
    if type == 'DR' or type == 'EM':
        loss = losses.MarginLoss(upper_margin=0.9, bottom_margin=0.1, down_weight=0.5)
        optimizer = keras.optimizers.Adam()
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = keras.optimizers.SGD(0.1)
    return loss, optimizer


def build_model(shape, num_out, params):
    inputs = keras.Input(shape=shape)
    model_name = build_model_name(params)
    probs, tensor_log = build(inputs, num_out,
                              params.routing.type,
                              params.routing.iter_num,
                              params.routing.temper,
                              params.caps.atoms,
                              params.model.resblock)
    model = keras.Model(inputs=inputs, outputs=probs, name=model_name)
    log_model = keras.Model(inputs=inputs, outputs=tensor_log.get_outputs(), name=model_name + '_log')
    tensor_log.set_model(log_model)
    loss, optimizer = get_loss_opt(params.routing.type)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[])
    model.summary()
    lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)
    lr_scheduler.set_model(model)
    callbacks = [lr_scheduler]
    model.callbacks = callbacks
    return model, tensor_log


def build(inputs, num_out, routing, iter_num, temper, atoms, resblock):
    log = utils.TensorLog()
    resblock = utils.parse_resblock(resblock)
    backbone = res_blocks.build_resnet_backbone(inputs=inputs, repetitions=resblock, layer_num=0,
                                                start_filters=16, arch='cifar',
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                                                version='v2')
    log.add_hist('backbone', backbone)
    if routing == 'avg':
        pool = keras.layers.GlobalAveragePooling2D()(backbone)
        output = keras.layers.Dense(num_out)(pool)
    elif routing == 'max':
        pool = keras.layers.GlobalMaxPooling2D()(backbone)
        output = keras.layers.Dense(num_out)(pool)
    elif routing == 'DR':
        child_pose, child_prob = layers.PrimaryCapsule(kernel_size=3,
                                                       strides=2,
                                                       padding='same',
                                                       groups=8,
                                                       use_bias=True,
                                                       atoms=8,
                                                       activation='squash',
                                                       kernel_initializer=kernel_initializer,
                                                       kernel_regularizer=kernel_regularizer)(backbone)
        log.add_hist('child_activation', child_prob)
        log.add_hist('child_pose', child_pose)
        transformed_caps = layers.CapsuleTransformDense(num_out=num_out,
                                                        out_atom=atoms,
                                                        share_weights=False,
                                                        initializer=keras.initializers.glorot_normal(),
                                                        regularizer=kernel_regularizer)(child_pose)
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
        child_pose, child_prob = layers.PrimaryCapsule(kernel_size=3,
                                                       strides=2,
                                                       padding='same',
                                                       groups=8,
                                                       use_bias=True,
                                                       atoms=8,
                                                       activation='sigmoid',
                                                       kernel_initializer=kernel_initializer,
                                                       kernel_regularizer=kernel_regularizer)(backbone)
        log.add_hist('child_activation', child_prob)
        log.add_hist('child_pose', child_pose)
        transformed_caps = layers.CapsuleTransformDense(num_out=num_out,
                                                        out_atom=atoms,
                                                        share_weights=False,
                                                        initializer=keras.initializers.glorot_normal(),
                                                        regularizer=kernel_regularizer)(child_pose)
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


def lr_schedule(epoch, lr):
    if epoch in [60, 80]:
        lr /= 10
    return lr


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
    elif params.task == 'attack':
        do_adv(os.getcwd())
    elif params.task == 'score':
        compute_entropies(os.getcwd())


def load_ckpt(model, model_dir, routing):
    loss, optimizer = get_loss_opt(routing)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[])
    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))


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


def get_model_dir(dataset, log='log', resblocks='333', routing='DR', iter_num=10, temper=1.0, atoms=16, idx=1):
    data_shape, num_out, flip, crop = get_input_set(dataset)
    model_dir = '{}/{}/{}_b{}_{}'.format(log, dataset, BASE_NAME, resblocks, routing)
    if routing == 'max' or routing == 'avg':
        model_dir += '_trial{}'.format(idx)
    else:
        model_dir += '_iter{}_temper{}_atoms{}_trial{}'.format(iter_num, temper, atoms, idx)
    if flip:
        model_dir += '_flip'
    if crop:
        model_dir += '_crop'

    if not os.path.exists(model_dir):
        raise Exception('model not exist:{}'.format(model_dir))
    return model_dir, data_shape, num_out, flip, crop


def load_model(data_shape, model_dir, num_out, routing, iter_num=10, temper=1.0, atoms=16, resblocks='333', input_norm=None):
    inputs = keras.Input(data_shape)
    probs, log = build(inputs=inputs if input_norm is None else layers.InputNorm(input_norm)(inputs),
                       num_out=num_out,
                       routing=routing,
                       iter_num=iter_num,
                       temper=temper,
                       atoms=atoms,
                       resblock=resblocks)
    model = keras.Model(inputs=inputs, outputs=probs, name='x')
    load_ckpt(model, model_dir, routing)
    return model


def evaluate_attack(epsilons, root='', log='log', dataset='kmnist', metric='acc', all_target=False,
                    method='FGSM', steps=10,
                    resblocks='333',
                    routing='DR', black_box=False, iter_num=10, temper=1.0, atoms=16, idx=1):
    model_dir, data_shape, num_out, flip, crop = get_model_dir(dataset, root+log, resblocks=resblocks,
                                                               routing=routing, iter_num=iter_num, temper=temper,
                                                               atoms=atoms, idx=idx)
    model = load_model(data_shape, model_dir, num_out, routing, iter_num, temper, atoms,
                       resblocks=resblocks, input_norm=get_norm_fn(dataset))
    if black_box:
        print('load black box source model')
        model_dir, data_shape, num_out, flip, crop = get_model_dir(dataset, root + log, resblocks=resblocks,
                                                                   routing=routing, iter_num=iter_num, temper=temper,
                                                                   atoms=atoms, idx=2)
        model_src = load_model(data_shape, model_dir, num_out, routing, iter_num, temper, atoms,
                               resblocks=resblocks, input_norm=get_norm_fn(dataset))
    else:
        model_src = model

    loss, _ = get_loss_opt(routing)
    _, test_set, info = data_input.build_dataset(dataset,
                                                 path=root + 'data',
                                                 parser_train=build_parse(dataset,
                                                                          flip=False,
                                                                          crop=False,
                                                                          is_train=True),
                                                 parser_test=build_parse(dataset,
                                                                         is_train=False,
                                                                         with_norm=False),
                                                 batch_size=512)

    acc_adv = keras.metrics.SparseCategoricalAccuracy(name='acc_adv')
    if metric == 'acc':
        results = attacks.evaluate_model_after_attacks(epsilons, acc_adv, test_set, model, loss, method=method, steps=steps, x_min=0, x_max=1, model_src=model_src)
    elif metric == 'success':
        if all_target:
            categories = [i for i in range(10)]
            results = attacks.evaluate_attacks_success_rate_all_target(epsilons, test_set, model, loss, categories, method=method, steps=steps, x_min=0, x_max=1, cost=True, model_src=model_src)
        else:
            results = attacks.evaluate_attacks_success_rate(epsilons, test_set, model, loss, method=method, steps=steps, x_min=0, x_max=1, model_src=model_src)
    return results


def do_adv(root):
    import time
    routing = 'EM'
    all_target = False
    tempers = [0.0, 20.0, 40.0, 60.0, 80.0]
    methods = ['PGD', 'BIM', 'FGSM']
    datasets = ['fashion_mnist', 'svhn_cropped', 'cifar10']
    black_box = False
    for dataset in datasets:
        print('dataset:', dataset)
        if dataset == 'cifar10':
            if all_target:
                epsilons = [0.05]
            else:
                epsilons = [0.01, 0.03, 0.06, 0.1]
        else:
            if all_target:
                epsilons = [0.1]
            else:
                epsilons = [0.1, 0.2, 0.3]
        for method in methods:
            print('method:', method)
            if routing == 'avg' or routing == 'max':
                tempers = [-1]
            for temper in tempers:
                print('temper:', temper)
                t1 = time.time()
                evaluate_attack(epsilons,
                                root=root,
                                log='log',
                                dataset=dataset,
                                metric='success',
                                all_target=all_target,
                                method=method,
                                steps=10,
                                routing=routing,
                                black_box=black_box,
                                iter_num=2,
                                temper=temper,
                                atoms=16,
                                idx=2)
                t2 = time.time()
                print('time:',t2-t1)


def compute_entropy(root,
                    log='log',
                    dataset='cifar10',
                    resblocks='333',
                    iter_num=2,
                    activated=True,
                    temper=10.0,
                    atoms=16,
                    routing='DR',
                    idx=1):
    import numpy as np
    model_dir, data_shape, num_out, flip, crop = get_model_dir(dataset, root+log, resblocks=resblocks,
                                                               routing=routing, iter_num=iter_num, temper=temper,
                                                               atoms=atoms, idx=idx)
    model = load_model(data_shape, model_dir, num_out, routing, iter_num, temper, atoms,
                       resblocks=resblocks, input_norm=None)
    _, test_set, info = data_input.build_dataset(dataset,
                                                 path=root + 'data',
                                                 parser_train=build_parse(dataset,
                                                                          flip=False,
                                                                          crop=False,
                                                                          is_train=True),
                                                 parser_test=build_parse(dataset,
                                                                         is_train=False),
                                                 batch_size=512)
    test_model = keras.Model(model.layers[0].input, [model.layers[69].output, model.layers[71].output])
    results = []
    for images, labels in test_set:
        (child_poses, child_probs), (parent_poses, parent_probs, cs) = test_model(images)
        c = cs[-1]
        if activated:
            entropy = activated_entropy(c, child_probs)
        else:
            entropy = coupling_entropy(c)
        results.append(entropy)
    results = np.concatenate(results, 0)
    mean = np.mean(results)
    std = np.std(results)
    print('{:.4}/{:.3}'.format(mean, std))


def compute_entropies(root):
    tempers = [0.0, 20.0, 40.0, 60.0, 80.0]
    datasets = ['cifar10', 'svhn_cropped', 'fashion_mnist']
    for dataset in datasets:
        print('dataset:', dataset)
        for temper in tempers:
            print('temper:{}'.format(temper))
            compute_entropy(root,
                            log='log',
                            dataset=dataset,
                            activated=False,
                            routing='EM',
                            iter_num=2,
                            temper=temper,
                            atoms=16,
                            idx=2)


if __name__ == "__main__":
    main()
