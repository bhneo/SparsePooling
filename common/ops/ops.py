from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

eps = 1e-10


def get_activation(name):
    if name == 'norm':
        return vector_norm
    elif name == 'norm_scale':
        return vector_norm_scale
    elif name == 'squash':
        return squash
    elif name == 'sum':
        return vector_sum
    elif name == 'mean':
        return vector_mean
    else:
        return None


def squash(inputs, axis=-1, ord="euclidean", name=None):
    name = "squashing" if name is None else name
    with tf.name_scope(name):
        norm = tf.norm(inputs, ord=ord, axis=axis, keepdims=True)
        norm_squared = tf.square(norm)
        scalar_factor = norm_squared / (1 + norm_squared)
        return scalar_factor * (inputs / (norm + eps)), scalar_factor


def vector_norm(inputs, axis=-1, ord="euclidean", name=None):
    name = "vector_norm" if name is None else name
    with tf.name_scope(name):
        norm = tf.norm(inputs, ord=ord, axis=axis, keepdims=True)
        return inputs / (norm + eps), norm


def vector_norm_scale(inputs, axis=-1, ord="euclidean", name=None):
    name = "vector_norm_scale" if name is None else name
    with tf.name_scope(name):
        norm = tf.norm(inputs, ord=ord, axis=axis, keepdims=True)
        return np.sqrt(inputs.get_shape().as_list()[axis])*inputs / (norm + eps), norm


def vector_sum(inputs, axis=-1, name=None):
    name = 'vector_sum' if name is None else name
    with tf.name_scope(name):
        return inputs, tf.reduce_sum(inputs, axis=axis, keepdims=True)


def vector_mean(inputs, axis=-1, name=None):
    name = 'vector_mean' if name is None else name
    with tf.name_scope(name):
        return inputs, tf.reduce_mean(inputs, axis=axis, keepdims=True)


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
