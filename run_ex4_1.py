import os
import argparse


if __name__ == "__main__":
    iterations = [1, 2, 3, 4, 5]
    for iteration in iterations:
        os.system(
            'python models/ex4_1.py --idx={} --task={} --routing={} --iter_num={} --temper={} --dataset={} --atoms={} --flip={} --crop={} --epochs={}'.format(
                1, 'train', 'DR', iteration, 1, 'cifar10', 16, True, True, 100
            ))

