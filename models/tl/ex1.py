import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from common.inputs.voc2010 import voc_parts
from common import layers, losses, utils, train, attacks
from common.ops.routing import activated_entropy, coupling_entropy

import numpy as np
import config


WEIGHT_DECAY = 1e-4
kernel_regularizer = keras.regularizers.l2(WEIGHT_DECAY)
kernel_initializer = keras.initializers.he_normal()
BASE_NAME = 'ex4_3'


def build_model_name(params):
    model_name = BASE_NAME
    model_name += '_{}'.format(params.model.backbone)
    model_name += '_fine{}'.format(params.model.fine)
    model_name += '_part{}'.format(params.caps.parts)
    model_name += '_{}'.format(params.routing.type)
    if params.routing.type == 'DR' or params.routing.type == 'EM':
        model_name += '_iter{}'.format(params.routing.iter_num)
        model_name += '_temper{}'.format(params.routing.temper)
        model_name += '_atoms{}'.format(params.caps.atoms)

    model_name += '_trial{}'.format(str(params.training.idx))
    model_name += '_bs{}'.format(str(params.training.batch_size))

    if params.dataset.flip:
        model_name += '_flip'
    if params.dataset.crop:
        model_name += '_crop'
    return model_name


def get_loss_opt(type):
    optimizer = keras.optimizers.Adam(0.0001)
    if type == 'DR' or type == 'EM':
        loss = losses.MarginLoss(sparse=False, upper_margin=0.9, bottom_margin=0.1, down_weight=0.5)
    else:
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    return loss, optimizer


def build_model(num_out, params):
    model_name = build_model_name(params)
    inputs, probs, tensor_log = build(num_out,
                                      params.model.backbone,
                                      params.model.fine,
                                      params.routing.type,
                                      params.routing.iter_num,
                                      params.routing.temper,
                                      params.caps.parts,
                                      params.caps.atoms
                                      )
    model = keras.Model(inputs=inputs, outputs=probs, name=model_name)
    log_model = keras.Model(inputs=inputs, outputs=tensor_log.get_outputs(), name=model_name + '_log')
    tensor_log.set_model(log_model)
    loss, optimizer = get_loss_opt(params.routing.type)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[])
    model.summary()
    model.callbacks = []
    return model, tensor_log


def build(num_out, backbone, fine, routing, iter_num, temper, parts, atoms):
    log = utils.TensorLog()
    if backbone == 'VGG16':
        in_shape = (224, 224, 3)
        base = keras.applications.VGG16(include_top=False, input_shape=in_shape)
    elif backbone == 'VGG19':
        in_shape = (224, 224, 3)
        base = keras.applications.VGG19(include_top=False, input_shape=in_shape)
    elif backbone == 'InceptionV3':
        in_shape = (299, 299, 3)
        base = keras.applications.InceptionV3(include_top=False, input_shape=in_shape)
    elif backbone == 'ResNet50':
        in_shape = (224, 224, 3)
        base = keras.applications.ResNet50(include_top=False, input_shape=in_shape)
    else:
        in_shape = (299, 299, 3)
        base = keras.applications.InceptionV3(include_top=False, input_shape=in_shape)

    layer_num = len(base.layers)
    for i, layer in enumerate(base.layers):
        if i < layer_num-fine:
            layer.trainable = False
        else:
            for w in layer.weights:
                if 'kernel' in w.name:
                    r = kernel_regularizer(w)
                    layer.add_loss(lambda: r)
    inputs = keras.Input(in_shape)
    features = base(inputs)
    interpretable = keras.layers.Conv2D(filters=parts,
                                        kernel_size=1,
                                        activation='relu',
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer)(features)

    shape = interpretable.get_shape().as_list()
    if routing == 'avg':
        pool = keras.layers.GlobalAveragePooling2D()(interpretable)
        output = keras.layers.Dense(num_out)(pool)
    elif routing == 'max':
        pool = keras.layers.GlobalMaxPooling2D()(interpretable)
        output = keras.layers.Dense(num_out)(pool)
    elif routing == 'DR':
        child_pose, child_prob = layers.CapsuleGroups(height=shape[1], width=shape[2], channel=shape[3],
                                                      atoms=16,
                                                      method='channel',
                                                      activation='squash')(interpretable)
        log.add_hist('child_activation', child_prob)
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
        output = parent_probs[-1]
    return inputs, output, log


def main():
    args, params = config.parse_args()
    if params.task == 'train':
        params.dataset.name = 'voc2010'
        if params.model.backbone == 'InceptionV3':
            data_shape = (299, 299, 3)
        else:
            data_shape = (224, 224, 3)
        train_set, test_set, info = voc_parts.build_dataset3(batch_size=params.training.batch_size,
                                                         shape=data_shape,
                                                         arch=params.model.backbone)
        model, tensor_log = build_model(num_out=info.features['label'].num_classes,
                                    params=params)

        trainer = train.Trainer(model, params, info, tensor_log, finetune=True, inference_label=False, max_save=1)
        trainer.metrics['accuracy'] = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        if args.train:
            trainer.fit(train_set, test_set)
        else:
            trainer.evaluate(test_set)
    elif params.task == 'attack':
        do_adv(os.getcwd())
    elif params.task == 'score':
        compute_entropies(os.getcwd())


def load_ckpt(model, model_dir):
    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[])
    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))


def get_model_dir(backbone, log='log', routing='avg', dataset='voc2010',
                  iter_num=None, temper=None, atoms=None,
                  finetune=0, parts=128, bs=32, idx=1):
    model_dir = '{}/{}/{}_{}_fine{}_part{}_{}'.format(log, dataset, BASE_NAME, backbone, finetune, parts, routing)
    if routing == 'DR' or routing == 'EM':
        model_dir += '_iter{}'.format(iter_num)
        model_dir += '_temper{}'.format(temper)
        model_dir += '_atoms{}'.format(atoms)

    model_dir += '_trial{}_bs{}_flip_crop'.format(idx, bs)

    if not os.path.exists(model_dir):
        raise Exception('model not exist:{}'.format(model_dir))
    return model_dir


def load_model(backbone, iter_num, temper, atoms=16,
               log='log', routing='DR',
               finetune=0, parts=128, bs=128, idx=1):
    data_shape = utils.get_shape(backbone)
    model_dir = get_model_dir(backbone=backbone,
                              log=log,
                              routing=routing,
                              finetune=finetune,
                              parts=parts,
                              bs=bs,
                              iter_num=iter_num,
                              temper=temper,
                              atoms=atoms,
                              idx=idx)
    inputs, probs, log = build(6, backbone, finetune, routing, iter_num, temper, parts, atoms)
    model = keras.Model(inputs=inputs, outputs=probs, name='x')
    load_ckpt(model, model_dir)
    return model, data_shape, model_dir


def evaluate_attack(epsilons, root='', log='log', backbone='InceptionV3', metric='acc', all_target=False,
                    method='FGSM', steps=10,
                    finetune=0, routing='DR', black_box=False, iter_num=10, temper=1.0, atoms=16, parts=128, bs=64, idx=1):
    model, data_shape, model_dir = load_model(log=root + log,
                                              backbone=backbone,
                                              routing=routing,
                                              iter_num=iter_num,
                                              temper=temper,
                                              atoms=atoms,
                                              parts=parts,
                                              bs=bs,
                                              finetune=finetune,
                                              idx=idx)
    if black_box:
        print('load black box source model')
        model_src, data_shape, model_dir = load_model(log=root + log,
                                                      backbone=backbone,
                                                      routing=routing,
                                                      iter_num=iter_num,
                                                      temper=temper,
                                                      atoms=atoms,
                                                      parts=parts,
                                                      bs=bs,
                                                      finetune=finetune,
                                                      idx=2)
    else:
        model_src = model

    loss, _ = get_loss_opt(routing)
    _, test_set, info = voc_parts.build_dataset3(root + 'data', batch_size=32, shape=data_shape)

    acc_adv = keras.metrics.CategoricalAccuracy(name='acc_adv')
    if metric == 'acc':
        results = attacks.evaluate_model_after_attacks(epsilons, acc_adv, test_set, model, loss, method=method, steps=steps, label_sparse=False, cost=True, model_src=model_src)
    elif metric == 'success':
        if all_target:
            categories = [i for i in range(6)]
            results = attacks.evaluate_attacks_success_rate_all_target(epsilons, test_set, model, loss, categories, method=method, steps=steps, label_sparse=False, cost=True, model_src=model_src)
        else:
            results = attacks.evaluate_attacks_success_rate(epsilons, test_set, model, loss, method=method, steps=steps, label_sparse=False, cost=True, model_src=model_src)
    return results


def do_adv(root):
    epsilons = [0.1, 0.2, 0.3]
    tempers = [0.0, 20.0, 40.0, 60.0, 80.0]
    parts_list = [128]
    all_target = False
    black_box = False
    methods = ['PGD', 'BIM', 'FGSM']
    backbones = ['InceptionV3']
    routing = 'DR'
    for backbone in backbones:
        print('backbone:', backbone)
        for parts in parts_list:
            print('parts:', parts)
            for method in methods:
                print('method:', method)
                if routing == 'avg' or routing == 'max':
                    tempers = [-1]
                for temper in tempers:
                    print('temper:', temper)
                    if all_target:
                        epsilons = [0.1]
                    evaluate_attack(epsilons,
                                    root=root,
                                    backbone=backbone,
                                    metric='success',
                                    all_target=all_target,
                                    method=method,
                                    steps=5,
                                    routing=routing,
                                    black_box=black_box,
                                    parts=parts,
                                    iter_num=2,
                                    temper=temper,
                                    atoms=16,
                                    bs=64,
                                    idx=1)


def compute_entropy(root,
                    backbone='InceptionV3',
                    iter_num=2,
                    activated=True,
                    temper=10.0,
                    atoms=16,
                    routing='DR',
                    finetune=0,
                    parts=128,
                    bs=32):
    model, data_shape, model_dir = load_model(log=root + 'log',
                                              backbone=backbone,
                                              iter_num=iter_num,
                                              temper=temper,
                                              atoms=atoms,
                                              routing=routing,
                                              finetune=finetune,
                                              parts=parts,
                                              bs=bs)
    train_set, test_set, info = voc_parts.build_dataset3(root + 'data', batch_size=32, shape=data_shape)
    test_model = keras.Model(model.layers[0].input, [model.layers[3].output, model.layers[5].output])
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
    for temper in tempers:
        print('temper:{}'.format(temper))
        compute_entropy(root,
                        backbone='InceptionV3',
                        iter_num=2,
                        temper=temper,
                        atoms=16,
                        routing='DR',
                        finetune=0,
                        parts=128,
                        bs=64)


if __name__ == "__main__":
    main()
