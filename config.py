from easydict import EasyDict
from common import utils

import argparse

params = EasyDict()

params.logdir = 'log'  # dir of logs
params.gpu = 1
params.task = 'train'

params.dataset = EasyDict()
params.dataset.name = 'cifar10'
params.dataset.target = 'cat'
params.dataset.flip = True
params.dataset.crop = True

params.training = EasyDict()
params.training.batch_size = 128
params.training.epochs = 163
params.training.steps = 9999999  # The number of training steps
params.training.lr_steps = [30000, 40000]
params.training.verbose = True
params.training.log_steps = 1000
params.training.idx = 1
params.training.momentum = 0.9
params.training.save_frequency = 10
params.training.log = True
params.training.whiten = 'zca'

params.routing = EasyDict()
params.routing.type = 'DR'
params.routing.iter_num = 3  # number of iterations in routing algorithm
params.routing.temper = 1  # the lambda in softmax

params.caps = EasyDict()
params.caps.parts = 64
params.caps.atoms = 8  # number of atoms in a capsule

params.model = EasyDict()
params.model.name = ''
params.model.layer_num = 20
params.model.resblock = '333'
params.model.backbone = 'VGG16'
params.model.fine = 3
params.model.pool = 'avg'
params.model.in_norm_fn = 'squash'
params.model.resnet = 'v2'


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
    parser.add_argument('--task', default='train', help='train or attack or score')
    parser.add_argument('--train', default=True, help='train of evaluate')
    parser.add_argument('--t_log', default=True, help='tensorboard log')
    parser.add_argument('--dataset', default=params.dataset.name, help='dataset config')
    parser.add_argument('--target', default=params.dataset.target, help='target')
    parser.add_argument('--flip', default=params.dataset.flip, help='dataset config')
    parser.add_argument('--crop', default=params.dataset.crop, help='dataset config')
    parser.add_argument('--idx', default=1, help='the index of trial')
    parser.add_argument('--epochs', default=params.training.epochs, help='the total training epochs')
    parser.add_argument('--batch', default=params.training.batch_size, help='the training batch_size')
    parser.add_argument('--steps', default=params.training.steps, help='the total training steps')
    parser.add_argument('--whiten', default=params.training.whiten, help='method to whiten')
    parser.add_argument('--log', default=params.logdir, help='directory to save log')
    parser.add_argument('--log_steps', default=params.training.log_steps, help='frequency to log by steps')
    parser.add_argument('--layer_num', default=params.model.layer_num, help='the number of layers')
    parser.add_argument('--resblock', default=params.model.resblock, help='res blocks')
    parser.add_argument('--resnet', default=params.model.resnet, help='resnet version')
    parser.add_argument('--backbone', default=params.model.backbone, help='backbones')
    parser.add_argument('--fine', default=params.model.fine, type=int, help='fine tune last layers')
    parser.add_argument('--pool', default=params.model.pool, help='pool')
    parser.add_argument('--in_norm_fn', default=params.model.in_norm_fn, help='')
    parser.add_argument('--routing', default=params.routing.type, help='')
    parser.add_argument('--temper', default=params.routing.temper, help='the lambda in softmax')
    parser.add_argument('--iter_num', default=params.routing.iter_num, help='the iter num of routing')
    parser.add_argument('--parts', default=params.caps.parts, help='number of parts filters')
    parser.add_argument('--atoms', default=params.caps.atoms, help='capsule atoms')
    arguments = parser.parse_args()
    build_params = build_config(arguments, params)
    return arguments, build_params


def build_config(args, build_params):
    build_params.gpu = args.gpu
    build_params.task = args.task
    build_params.logdir = args.log
    build_params.dataset.name = args.dataset
    build_params.dataset.flip = utils.str2bool(args.flip)
    build_params.dataset.crop = utils.str2bool(args.crop)
    build_params.dataset.target = args.target
    build_params.training.log_steps = int(args.log_steps)
    build_params.training.idx = args.idx
    build_params.training.epochs = int(args.epochs)
    build_params.training.batch_size = int(args.batch)
    build_params.training.steps = int(args.steps)
    build_params.training.log = utils.str2bool(args.t_log)
    build_params.training.whiten = args.whiten
    build_params.model.layer_num = int(args.layer_num)
    build_params.model.resblock = args.resblock
    build_params.model.pool = args.pool
    build_params.model.backbone = args.backbone
    build_params.model.fine = args.fine
    build_params.model.in_norm_fn = args.in_norm_fn
    build_params.model.resnet = args.resnet
    build_params.routing.type = args.routing
    build_params.routing.temper = float(args.temper)
    build_params.routing.iter_num = int(args.iter_num)
    build_params.caps.parts = int(args.parts)
    build_params.caps.atoms = int(args.atoms)
    return build_params
