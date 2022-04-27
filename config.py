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
params.training.momentum = 0.9
params.training.save_frequency = 10
params.training.log = True
params.training.whiten = 'zca'

params.model = EasyDict()
params.model.name = ''
params.model.layer_num = 20
params.model.resblock = '333'
params.model.backbone = 'VGG16'
params.model.fine = 3
params.model.pool = 'avg'
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
    build_params.model.resnet = args.resnet
    return build_params
