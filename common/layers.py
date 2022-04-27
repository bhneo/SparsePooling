import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer

eps = 1e-10


class InputNorm(Layer):
    def __init__(self,
                 norm_fn,
                 **kwargs):
        super(InputNorm, self).__init__(**kwargs)
        self.norm_fn = norm_fn

    def call(self, inputs, **kwargs):
        return self.norm_fn(inputs)


def get_cross_mul(inputs):
    n = inputs.get_shape().as_list()[-2]
    outputs = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                outputs += inputs[i] * inputs[j]
    outputs /= (2 * n)
    return outputs


def batch_2d(inputs, kernel_size, strides, name=None):
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(strides, tuple):
        strides = (strides, strides)
    name = "batch_to_pool" if name is None else name
    with tf.name_scope(name):
        height, width = inputs.get_shape().as_list()[1:3]
        h_offsets = [[(h + k) for k in range(0, kernel_size[0])] for h in range(0, height + 1 - kernel_size[0], strides[0])]
        w_offsets = [[(w + k) for k in range(0, kernel_size[1])] for w in range(0, width + 1 - kernel_size[1], strides[1])]
        patched = tf.gather(inputs, h_offsets, axis=1)
        patched = tf.gather(patched, w_offsets, axis=3)
        perm = [0, 1, 3, 2, 4, 5]
        patched = tf.transpose(patched, perm=perm)
        shape = patched.get_shape().as_list()
        shape = [-1] + shape[1:3] + [np.prod(shape[3:-1]), shape[-1]]
        patched = tf.reshape(patched, shape=shape)
    return patched


def test_batch_2d():
    x = tf.random.normal([128, 32, 32, 3])
    x = batch_2d(x, 2, 2)
    print(x.shape)

