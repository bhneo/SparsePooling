from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

eps = 1e-10


def center(inputs, axis):
    m = tf.reduce_mean(inputs, axis=axis, keepdims=True)
    f = inputs - m
    return m, f


def get_decomposition(decomposition, iter_num, eplison):
    if decomposition == 'cholesky':
        def get_inv_sqrt(ff, dim):
            with tf.device('/cpu:0'):
                sqrt = tf.linalg.cholesky(ff)
            inv_sqrt = tf.linalg.triangular_solve(sqrt, tf.eye(dim))
            return inv_sqrt
    elif decomposition == 'zca':
        def get_inv_sqrt(ff, dim):
            with tf.device('/cpu:0'):
                eig, rotation, _ = tf.linalg.svd(ff, full_matrices=True)
            eig += eplison
            eig = tf.linalg.diag(tf.pow(eig, -0.5))
            inv_sqrt = tf.matmul(tf.matmul(rotation, eig), rotation, transpose_b=True)
            return inv_sqrt
    elif decomposition == 'pca':
        def get_inv_sqrt(ff, dim):
            with tf.device('/cpu:0'):
                eig, rotation, _ = tf.linalg.svd(ff, full_matrices=True)
            eig += eplison
            eig = tf.linalg.diag(tf.pow(eig, -0.5))
            inv_sqrt = tf.matmul(rotation, eig, transpose_a=True)
            return inv_sqrt
    elif decomposition == 'iter_norm':
        def get_inv_sqrt(ff, dim):
            trace = tf.linalg.trace(ff)
            trace = tf.expand_dims(trace, [-1])
            trace = tf.expand_dims(trace, [-1])
            sigma_norm = ff / trace

            projection = tf.eye(dim)
            for i in range(iter_num):
                projection = (3 * projection - projection * projection * projection * sigma_norm) / 2

            return projection / tf.sqrt(trace)
    else:
        assert False
    return get_inv_sqrt
