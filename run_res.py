import os
import argparse


def run(args):
    tempers = [1,5,10,20,60]
    if args.gpu == '0':
        for temper in tempers:
            os.system(
                    'python models/res/ex1.py --idx={} --task={} --routing={} --in_fn={} --out_fn={} --in_norm={}\
                    --iter_num={} --temper={} --dataset={} --atoms={} --flip={} --crop={} --epochs={}'.format(
                        1, 'train', 'NR', None, None, None, 1, temper, 'cifar10', 16, True, True, 100
                    ))
    elif args.gpu == '1':
        for temper in tempers:
            os.system(
                'python models/res/ex1.py --idx={} --task={} --routing={} --in_fn={} --out_fn={} --in_norm={}\
                --iter_num={} --temper={} --dataset={} --atoms={} --flip={} --crop={} --epochs={}'.format(
                    1, 'train', 'NR', None, None, None, 1, temper, 'cifar10', 64, True, True, 100
                ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--iter_num', default=1, type=int, help='iter_num')
    parser.add_argument('--temper', default=1.0, type=float, help='temper')
    parser.add_argument('--atoms', default=16, type=int, help='iter_num')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--flip', default=False, help='flip')
    parser.add_argument('--crop', default=True, help='crop')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    arguments = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    run(arguments)