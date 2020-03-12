import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, math_ops, state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.framework import ops

from common import utils
from common.ops import transformation
from common.ops import ops as custom_ops
from common.ops.em_routing import em_routing
from common.ops.routing import dynamic_routing, norm_routing
from common.ops.routing import activated_entropy
from config import params as cfg

eps = 1e-10


class Activation(Layer):
    def __init__(self,
                 activation='squash',
                 with_prob=False,
                 **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.activation_fn = custom_ops.get_activation(activation)
        self.with_prob = with_prob

    def call(self, inputs, **kwargs):
        if self.activation_fn:
            pose, prob = self.activation_fn(inputs, axis=-1)
        else:
            pose, prob = inputs, None
        if self.with_prob:
            return pose, prob
        else:
            return pose


class InputNorm(Layer):
    def __init__(self,
                 norm_fn,
                 **kwargs):
        super(InputNorm, self).__init__(**kwargs)
        self.norm_fn = norm_fn

    def call(self, inputs, **kwargs):
        return self.norm_fn(inputs)


class PrimaryCapsule(Layer):
    def __init__(self,
                 kernel_size,
                 strides,
                 use_bias=False,
                 conv_caps=False,
                 padding='valid',
                 groups=32,
                 atoms=8,
                 activation='squash',
                 bn=False,
                 kernel_initializer=keras.initializers.glorot_normal(),
                 kernel_regularizer=None,
                 **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.atoms = atoms
        self.conv_caps = conv_caps
        if bn:
            self.bn = keras.layers.BatchNormalization()
        else:
            self.bn = None
        self.conv = keras.layers.Conv2D(filters=self.groups * self.atoms,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        use_bias=use_bias,
                                        activation=None,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer)
        if activation == 'sigmoid':
            self.activation_fn = keras.layers.Conv2D(filters=self.groups * 1,
                                                     kernel_size=self.kernel_size,
                                                     strides=self.strides,
                                                     padding=self.padding,
                                                     use_bias=use_bias,
                                                     activation='sigmoid',
                                                     kernel_initializer=kernel_initializer,
                                                     kernel_regularizer=kernel_regularizer)
        else:
            self.activation_fn = custom_ops.get_activation(activation)

    def call(self, inputs, **kwargs):
        pose = self.conv(inputs)
        if self.bn is not None:
            pose = self.bn(pose)
        pose_shape = pose.get_shape().as_list()
        if self.conv_caps:
            pose = tf.reshape(pose, shape=[-1, pose_shape[1], pose_shape[2], self.groups, self.atoms])
        else:
            pose = tf.reshape(pose, shape=[-1, pose_shape[1]*pose_shape[2]*self.groups, self.atoms])
        if isinstance(self.activation_fn, keras.layers.Conv2D):
            prob = self.activation_fn(inputs)
            if self.conv_caps:
                prob = tf.reshape(prob, shape=[-1, pose_shape[1], pose_shape[2], self.groups, 1])
            else:
                prob = tf.reshape(prob, shape=[-1, pose_shape[1] * pose_shape[2] * self.groups, 1])
        else:
            pose, prob = self.activation_fn(pose, axis=-1)
        return pose, prob


class CapsuleTransformDense(Layer):
    def __init__(self,
                 num_out,
                 out_atom,
                 share_weights=False,
                 matrix=False,
                 initializer=keras.initializers.glorot_normal(),
                 regularizer=None,
                 **kwargs):
        super(CapsuleTransformDense, self).__init__(**kwargs)
        self.num_out = num_out
        self.out_atom = out_atom
        self.share_weights = share_weights
        self.matrix = matrix
        self.wide = None
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        in_atom = input_shape[-1]
        in_num = input_shape[-2]
        if self.matrix:
            self.wide = int(np.sqrt(in_atom))
        if self.share_weights:
            if self.wide:
                self.kernel = self.add_weight(
                    name='capsule_kernel',
                    shape=(1, self.num_out, self.wide, self.wide),
                    initializer=self.initializer,
                    regularizer=self.regularizer,
                    trainable=True)
                self.kernel = tf.tile(self.kernel, [in_num, 1, 1, 1])
            else:
                self.kernel = self.add_weight(
                    name='capsule_kernel',
                    shape=(1, in_atom,
                           self.num_out * self.out_atom),
                    initializer=self.initializer,
                    regularizer=self.regularizer,
                    trainable=True)
                self.kernel = tf.tile(self.kernel, [in_num, 1, 1])
        else:
            if self.wide:
                self.kernel = self.add_weight(
                    name='capsule_kernel',
                    shape=(in_num, self.num_out, self.wide, self.wide),
                    initializer=self.initializer,
                    regularizer=self.regularizer,
                    trainable=True)
            else:
                self.kernel = self.add_weight(
                    name='capsule_kernel',
                    shape=(in_num, in_atom,
                           self.num_out * self.out_atom),
                    initializer=self.initializer,
                    regularizer=self.regularizer,
                    trainable=True)

    def call(self, inputs, **kwargs):
        in_shape = inputs.get_shape().as_list()
        in_shape[0] = -1
        if self.wide:
            # [bs, in_num, in_atom] -> [bs, in_num, wide, wide]
            inputs = tf.reshape(inputs, in_shape[:-1]+[self.wide, self.wide])
            # [bs, in_num, a, b]  X  [in_num, out_num, b, c]
            # -> [bs, in_num, out_num, a, c]
            outputs = transformation.matrix_capsule_element_wise(inputs, self.kernel, self.num_out)
            outputs = tf.reshape(outputs, in_shape[:-1] + [self.num_out] + [in_shape[-1]])
        else:
            # [bs, in_num, in_atom] X [in_num, in_atom, out_num*out_atom]
            #  -> [bs, in_num, out_num, out_atom]
            outputs = transformation.matmul_element_wise(inputs, self.kernel, self.num_out, self.out_atom)
        return outputs


class CapsuleTransformConv(Layer):
    def __init__(self,
                 kernel_size,
                 stride,
                 filter,
                 atom,
                 initializer=keras.initializers.glorot_normal(),
                 regularizer=None,
                 **kwargs):
        super(CapsuleTransformConv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.filter = filter
        self.atom = atom
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        # inputs [bs, height, width, channel, in_atom]
        in_channel = input_shape[-2]
        in_atom = input_shape[-1]

        self.matrix = self.add_weight(
            name='capsule_kernel',
            shape=(self.kernel_size*self.kernel_size*in_channel, in_atom,
                   self.filter*self.atom),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True)

    def call(self, inputs, **kwargs):
        # inputs [bs, height, width, channel, in_atom]
        inputs_tile, _ = utils.kernel_tile(inputs, self.kernel_size, self.stride)
        # tile [bs, out_height, out_width, kernel*kernel*channel, in_atom]
        outputs = transformation.matmul_element_wise(inputs_tile, self.matrix, self.filter, self.atom)
        # [bs, out_height, out_width, kernel*kernel*channel, out_num, out_atom]
        return outputs


class CapsuleGroups(Layer):
    def __init__(self,
                 height,
                 width,
                 channel,
                 atoms,
                 norm=None,
                 activation=None,
                 initializer=keras.initializers.glorot_normal(),
                 regularizer=None,
                 log=None,
                 **kwargs):
        super(CapsuleGroups, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.channel = channel
        self.atoms = atoms
        self.groups = self.channel // self.atoms
        self.activation = activation
        self.norm = norm
        self.log = log
        if activation == 'sigmoid':
            self.conv = keras.layers.Conv2D(filters=self.groups,
                                            kernel_size=1,
                                            activation='sigmoid',
                                            kernel_initializer=initializer,
                                            kernel_regularizer=regularizer)

    def call(self, inputs, **kwargs):
        groups = tf.reshape(inputs, shape=[-1, self.height * self.width, self.groups, self.atoms])
        prob = None
        if self.norm is None:
            pose = groups
        else:
            pose, prob = custom_ops.get_activation(self.norm)(groups)
        if self.activation is not None and self.activation != self.norm:
            if self.activation == 'sigmoid':
                prob = self.conv(inputs)
                prob = tf.reshape(prob, [-1, self.height * self.width, self.groups, 1])
            else:
                _, prob = custom_ops.get_activation(self.activation)(groups)

        if self.log:
            self.log.add_hist('child_pose', pose)
            if prob:
                self.log.add_hist('child_activation', prob)
        if prob is None:
            return pose
        else:
            return pose, prob


class RoutingPooling(Layer):
    def __init__(self,
                 kernel_size,
                 strides,
                 atoms,
                 in_norm=True,
                 num_routing=2,
                 temper=1.0,
                 **kwargs):
        super(RoutingPooling, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.atoms = atoms
        self.in_norm = in_norm
        self.num_routing = num_routing
        self.temper = temper

    def call(self, inputs, **kwargs):
        patched = batch_2d(inputs, self.kernel_size, self.strides)
        patched_shape = patched.get_shape().as_list()
        patched = tf.reshape(patched,
                             [-1] + patched_shape[1:4] + [patched_shape[4] // self.atoms, self.atoms])
        patched = tf.transpose(patched, perm=[0, 1, 2, 4, 3, 5])
        patched = tf.expand_dims(patched, axis=-2)

        pose, _ = dynamic_routing(patched,
                                  num_routing=self.num_routing,
                                  softmax_in=True,
                                  temper=self.temper,
                                  activation='norm')
        pose = tf.reshape(pose, [-1] + patched_shape[1:3] + [patched_shape[4]])
        return pose


class DynamicRouting(Layer):
    def __init__(self,
                 num_routing=3,
                 softmax_in=False,
                 temper=1.0,
                 activation='squash',
                 pooling=False,
                 single=False,
                 with_b=False,
                 log=None,
                 **kwargs):
        super(DynamicRouting, self).__init__(**kwargs)
        self.num_routing = num_routing
        self.softmax_in = softmax_in
        self.temper = temper
        self.activation = activation
        self.pooling = pooling
        self.single = single
        self.with_b = with_b
        self.log = log

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            (predictions, child_prob) = inputs
        else:
            predictions = inputs
            child_prob = None
        if self.pooling:
            predictions = tf.expand_dims(predictions, -2)
        poses, probs, bs, cs = dynamic_routing(predictions,
                                               num_routing=self.num_routing,
                                               softmax_in=self.softmax_in,
                                               temper=self.temper,
                                               activation=self.activation)
        for i in range(self.num_routing):
            if self.log:
                if self.with_b:
                    self.log.add_hist('b{}'.format(i + 1), bs[i])
                self.log.add_hist('c{}'.format(i + 1), cs[i])
                entropy = activated_entropy(cs[i], child_prob)
                self.log.add_hist('rEntropy{}'.format(i + 1), entropy)
                self.log.add_scalar('rAvgEntropy{}'.format(i + 1), tf.reduce_mean(entropy))
            poses[i] = tf.squeeze(poses[i], axis=-3)
            probs[i] = tf.squeeze(probs[i], axis=[-3, -1])
            # if self.single:
            #     poses[i] = tf.split(poses[i], [1, 1], -2)[0]
            #     probs[i] = tf.split(probs[i], [1, 1], -1)[0]
        if self.with_b is True:
            return poses, probs, bs, cs
        return poses, probs, cs


class NormRouting(Layer):
    def __init__(self,
                 num_routing=1,
                 softmax_in=False,
                 temper=1.0,
                 activation='squash',
                 pooling=False,
                 log=None,
                 **kwargs):
        super(NormRouting, self).__init__(**kwargs)
        self.num_routing = num_routing
        self.softmax_in = softmax_in
        self.temper = temper
        self.activation = activation
        self.pooling = pooling
        self.log = log

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            (predictions, child_prob) = inputs
        else:
            predictions = inputs
            child_prob = None
        if self.pooling:
            predictions = tf.expand_dims(predictions, -2)
        poses, probs, bs, cs = norm_routing(predictions,
                                            num_routing=self.num_routing,
                                            softmax_in=self.softmax_in,
                                            temper=self.temper,
                                            activation=self.activation)
        for i in range(self.num_routing):
            if self.log:
                self.log.add_hist('c{}'.format(i + 1), cs[i])
                entropy = activated_entropy(cs[i], child_prob)
                self.log.add_hist('rEntropy{}'.format(i + 1), entropy)
                self.log.add_scalar('rAvgEntropy{}'.format(i + 1), tf.reduce_mean(entropy))
            poses[i] = tf.squeeze(poses[i], axis=-3)
            if probs:
                if probs[i] is not None:
                    probs[i] = tf.squeeze(probs[i], axis=[-3, -1])
        if len(probs) == 0:
            return poses, cs
        else:
            return poses, probs, cs


class EMRouting(Layer):
    def __init__(self,
                 num_routing=3,
                 temper=1.0,
                 softmax_in=False,
                 log=None,
                 **kwargs):
        super(EMRouting, self).__init__(**kwargs)
        self.num_routing = num_routing
        self.temper = temper
        self.softmax_in = softmax_in
        self.log = log

    def build(self, input_shape):
        # ----- Betas -----#

        """
        # Initialization from Jonathan Hui [1]:
        beta_v_hui = tf.get_variable(
          name='beta_v',
          shape=[1, 1, 1, o],
          dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        beta_a_hui = tf.get_variable(
          name='beta_a',
          shape=[1, 1, 1, o],
          dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())

        # AG 21/11/2018:
        # Tried to find std according to Hinton's comments on OpenReview
        # https://openreview.net/forum?id=HJWLfGWRb&noteId=r1lQjCAChm
        # Hinton: "We used truncated_normal_initializer and set the std so that at the
        # start of training half of the capsules in each layer are active and half
        # inactive (for the Primary Capsule layer where the activation is not computed
        # through routing we use different std for activation convolution weights &
        # for pose parameter convolution weights)."
        #
        # std beta_v seems to control the spread of activations
        # To try and achieve what Hinton said about half active and half not active,
        # I change the std values and check the histogram/distributions in
        # Tensorboard
        # to try and get a good spread across all values. I couldn't get this working
        # nicely.
        beta_v_hui = slim.model_variable(
          name='beta_v',
          shape=[1, 1, 1, 1, o, 1],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=10.0))
        """
        o = input_shape[0].as_list()[-2]  # out caps
        self.beta_a = self.add_weight(name='beta_a',
                                      shape=[1, 1, o, 1],
                                      dtype=tf.float32,
                                      initializer=tf.keras.initializers.TruncatedNormal(mean=-1000.0, stddev=500.0))

        # AG 04/10/2018: using slim.variable to create instead of tf.get_variable so
        # that they get correctly placed on the CPU instead of GPU in the multi-gpu
        # version.
        # One beta per output capsule type
        # (N, i, o, atom)
        self.beta_v = self.add_weight(name='beta_v',
                                      shape=[1, 1, o, 1],
                                      dtype=tf.float32,
                                      initializer=tf.keras.initializers.GlorotNormal(),
                                      regularizer=None)

    def call(self, inputs, **kwargs):
        # votes (bs, in, out, atom)
        # activations (bs, in, 1)
        if isinstance(inputs, tuple) and len(inputs) == 2:
            (predictions, child_prob) = inputs
        else:
            predictions = inputs
            child_prob = None
        poses, probs, cs = em_routing(predictions,
                                      child_prob,
                                      self.beta_a,
                                      self.beta_v,
                                      self.num_routing,
                                      self.softmax_in,
                                      self.temper,
                                      final_lambda=0.01,
                                      epsilon=1e-9,
                                      spatial_routing_matrix=[[1]],
                                      log=self.log)

        for i in range(self.num_routing):
            if self.log:
                self.log.add_hist('c{}'.format(i + 1), cs[i])
                entropy = activated_entropy(cs[i], child_prob)
                self.log.add_hist('rEntropy{}'.format(i + 1), entropy)
                self.log.add_scalar('rAvgEntropy{}'.format(i + 1), tf.reduce_mean(entropy))
            probs[i] = tf.squeeze(probs[i], axis=[-1])
        return poses, probs, cs


class CapsDecorelationNormalization(Layer):
    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-3,
                 decomposition='iter_norm',
                 iter_num=5,
                 affine=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 moving_mean_initializer='zeros',
                 moving_whiten_initializer='identity',
                 **kwargs):
        assert decomposition in ['zca', 'pca', 'iter_norm']
        super(CapsDecorelationNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_whiten_initializer = initializers.get(moving_whiten_initializer)
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.affine = affine
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.atom = 0
        self.caps_out = 0
        self.caps_in = 0

    def matrix_initializer(self, shape, dtype=tf.float32, partition_info=None):
        moving_conv = tf.eye(shape[-1], dtype=dtype)
        moving_conv = tf.reshape(moving_conv, [1, shape[-1], shape[-1]])
        moving_conv = tf.tile(moving_conv, [shape[0], 1, 1])
        return moving_conv

    def build(self, input_shape):
        # shape should be (..., caps_in, caps_out, atom)
        input_shape_list = input_shape.as_list()
        self.atom = input_shape_list[-1]
        self.caps_out = input_shape_list[-2]
        self.caps_in = input_shape_list[-3]
        if self.atom is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.moving_mean = self.add_weight(shape=(1, self.caps_in, 1, self.atom),
                                           name='moving_mean',
                                           synchronization=tf_variables.VariableSynchronization.ON_READ,
                                           initializer=self.moving_mean_initializer,
                                           trainable=False,
                                           aggregation=tf_variables.VariableAggregation.MEAN)
        self.moving_matrix = self.add_weight(shape=(self.caps_in, self.atom, self.atom),
                                             name='moving_matrix',
                                             synchronization=tf_variables.VariableSynchronization.ON_READ,
                                             initializer=self.matrix_initializer,
                                             trainable=False,
                                             aggregation=tf_variables.VariableAggregation.MEAN)

        if self.affine:
            param_shape = (1, self.caps_in, 1, self.atom)
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            self.beta = None

        self.built = True

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def call(self, inputs, training=None):
        # (bs, caps_in, caps_out, atom)
        mean, centered = custom_ops.center(inputs, [0, 2])
        norm_dim = cfg.training.batch_size*self.caps_out
        centered = tf.transpose(centered, [1, 0, 2, 3])  # to [caps_in, caps_out, bs, atom]
        centered = tf.reshape(centered, [self.caps_in, -1, self.atom])
        mean = K.in_train_phase(mean, self.moving_mean)
        get_inv_sqrt = custom_ops.get_decomposition(self.decomposition, self.iter_num, self.epsilon)

        def train():
            sigma = tf.matmul(centered, centered, transpose_a=True)
            sigma /= (tf.cast(norm_dim, tf.float32) - 1.)
            whiten_matrix = get_inv_sqrt(sigma, self.atom)
            self.add_update([self._assign_moving_average(self.moving_mean, mean, self.momentum, None),
                             self._assign_moving_average(self.moving_matrix, whiten_matrix, self.momentum, None)],
                            inputs)
            return whiten_matrix

        def test():
            moving_matrix = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.atom) * self.epsilon
            return moving_matrix

        inv_sqrt = K.in_train_phase(train, test)
        decorelated = tf.matmul(centered, inv_sqrt)
        decorelated = tf.reshape(decorelated, [self.caps_in, -1, self.caps_out, self.atom])
        decorelated = tf.transpose(decorelated, [1, 0, 2, 3])  # to (bs, caps_in, caps_out, atom)

        if self.gamma is not None:
            decorelated *= self.gamma
        if self.beta is not None:
            decorelated += self.beta
        return decorelated

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_matrix_initializer': initializers.serialize(self.matrix_initializer)
        }
        base_config = super(CapsDecorelationNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsDecorelationNormalization2(Layer):  # axis [bs, atom]
    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-3,
                 decomposition='iter_norm',
                 iter_num=5,
                 affine=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 moving_mean_initializer='zeros',
                 moving_whiten_initializer='identity',
                 **kwargs):
        assert decomposition in ['zca', 'pca', 'iter_norm', 'cholesky']
        super(CapsDecorelationNormalization2, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_whiten_initializer = initializers.get(moving_whiten_initializer)
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.affine = affine
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.atom = 0
        self.caps_out = 0
        self.caps_in = 0

    def build(self, input_shape):
        # shape should be (..., caps_in, caps_out, atom)
        input_shape_list = input_shape.as_list()
        self.atom = input_shape_list[-1]
        self.caps_out = input_shape_list[-2]
        self.caps_in = input_shape_list[-3]
        if self.atom is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if self.affine:
            param_shape = (1, self.caps_in, 1, 1)
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            self.beta = None

        self.built = True

    def call(self, inputs, training=None):
        # (bs, caps_in, caps_out, atom)
        mean, centered = custom_ops.center(inputs, [0, 2, 3])
        norm_dim = self.caps_out*self.atom*cfg.training.batch_size
        # centered = tf.reshape(centered, [-1, self.atom])
        centered = tf.transpose(centered, [0, 2, 3, 1])  # to [caps_out, bs, atom, caps_in]
        centered = tf.reshape(centered, [-1, self.caps_in])
        get_inv_sqrt = custom_ops.get_decomposition(self.decomposition, self.iter_num, self.epsilon)

        def decom():
            sigma = tf.matmul(centered, centered, transpose_a=True)
            sigma /= (tf.cast(norm_dim, tf.float32) - 1.)
            whiten_matrix = get_inv_sqrt(sigma, self.caps_in)
            return whiten_matrix

        inv_sqrt = decom()
        decorelated = tf.matmul(centered, inv_sqrt)
        decorelated = tf.reshape(decorelated, [-1, self.caps_out, self.atom, self.caps_in])
        decorelated = tf.transpose(decorelated, [0, 3, 1, 2])  # to (bs, caps_in, caps_out, atom)

        if self.gamma is not None:
            decorelated *= self.gamma
        if self.beta is not None:
            decorelated += self.beta
        return decorelated

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_matrix_initializer': initializers.serialize(self.matrix_initializer)
        }
        base_config = super(CapsDecorelationNormalization2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsDecorelationNormalization3(Layer):  # axis [bs, atom]
    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-3,
                 decomposition='iter_norm',
                 iter_num=5,
                 affine=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 moving_mean_initializer='zeros',
                 moving_whiten_initializer='identity',
                 **kwargs):
        assert decomposition in ['zca', 'pca', 'iter_norm', 'cholesky']
        super(CapsDecorelationNormalization3, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_whiten_initializer = initializers.get(moving_whiten_initializer)
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.affine = affine
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.channel = 0
        self.height = 0
        self.width = 0

    def matrix_initializer(self, shape, dtype=tf.float32, partition_info=None):
        moving_conv = tf.eye(shape[-1], dtype=dtype)
        return moving_conv

    def build(self, input_shape):
        # shape should be (..., caps_in, caps_out, atom)
        input_shape_list = input_shape.as_list()
        self.channel = input_shape_list[-1]
        self.height = input_shape_list[-2]
        self.width = input_shape_list[-3]

        self.moving_mean = self.add_weight(shape=(1, self.height, self.width, 1),
                                           name='moving_mean',
                                           synchronization=tf_variables.VariableSynchronization.ON_READ,
                                           initializer=self.moving_mean_initializer,
                                           trainable=False,
                                           aggregation=tf_variables.VariableAggregation.MEAN)
        self.moving_matrix = self.add_weight(shape=(self.height*self.width, self.height*self.width),
                                             name='moving_matrix',
                                             synchronization=tf_variables.VariableSynchronization.ON_READ,
                                             initializer=self.matrix_initializer,
                                             trainable=False,
                                             aggregation=tf_variables.VariableAggregation.MEAN)

        if self.affine:
            param_shape = (1, self.height, self.width, 1)
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            self.beta = None

        self.built = True

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def call(self, inputs, training=None):
        # (bs, height, width, channel)
        mean, centered = custom_ops.center(inputs, [0, 3])
        norm_dim = self.channel*cfg.training.batch_size
        centered = tf.transpose(centered, [0, 3, 1, 2])
        centered = tf.reshape(centered, [-1, self.height*self.width])
        mean = K.in_train_phase(mean, self.moving_mean)
        get_inv_sqrt = custom_ops.get_decomposition(self.decomposition, self.iter_num, self.epsilon)

        def train():
            sigma = tf.matmul(centered, centered, transpose_a=True)
            sigma /= (tf.cast(norm_dim, tf.float32) - 1.)
            whiten_matrix = get_inv_sqrt(sigma, self.width*self.height)
            self.add_update([self._assign_moving_average(self.moving_mean, mean, self.momentum, None),
                             self._assign_moving_average(self.moving_matrix, whiten_matrix, self.momentum, None)],
                            inputs)
            return whiten_matrix

        def test():
            moving_matrix = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.height*self.width) * self.epsilon
            return moving_matrix

        inv_sqrt = K.in_train_phase(train, test)
        decorelated = tf.matmul(centered, inv_sqrt)
        decorelated = tf.reshape(decorelated, [-1, self.channel, self.height, self.width])
        decorelated = tf.transpose(decorelated, [0, 2, 3, 1])  # to (bs, caps_in, caps_out, atom)

        if self.gamma is not None:
            decorelated *= self.gamma
        if self.beta is not None:
            decorelated += self.beta
        return decorelated

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_matrix_initializer': initializers.serialize(self.matrix_initializer)
        }
        base_config = super(CapsDecorelationNormalization3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsDecorelationNormalization4(Layer):
    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-3,
                 decomposition='iter_norm',
                 iter_num=5,
                 affine=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 moving_mean_initializer='zeros',
                 moving_whiten_initializer='identity',
                 **kwargs):
        assert decomposition in ['zca', 'pca', 'iter_norm']
        super(CapsDecorelationNormalization4, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_whiten_initializer = initializers.get(moving_whiten_initializer)
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.affine = affine
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.atom = 0
        self.caps_in = 0

    def matrix_initializer(self, shape, dtype=tf.float32, partition_info=None):
        moving_conv = tf.eye(shape[-1], dtype=dtype)
        moving_conv = tf.expand_dims(moving_conv, 0)
        moving_conv = tf.tile(moving_conv, [shape[0], 1, 1])
        return moving_conv

    def build(self, input_shape):
        # shape should be (..., caps_in, atom)
        input_shape_list = input_shape.as_list()
        self.atom = input_shape_list[-1]
        self.caps_in = input_shape_list[-2]
        if self.atom is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.moving_mean = self.add_weight(shape=(1, self.caps_in, self.atom),
                                           name='moving_mean',
                                           synchronization=tf_variables.VariableSynchronization.ON_READ,
                                           initializer=self.moving_mean_initializer,
                                           trainable=False,
                                           aggregation=tf_variables.VariableAggregation.MEAN)
        self.moving_matrix = self.add_weight(shape=(self.caps_in, self.atom, self.atom),
                                             name='moving_matrix',
                                             synchronization=tf_variables.VariableSynchronization.ON_READ,
                                             initializer=self.matrix_initializer,
                                             trainable=False,
                                             aggregation=tf_variables.VariableAggregation.MEAN)

        if self.affine:
            param_shape = (1, self.caps_in, self.atom)
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            self.beta = None

        self.built = True

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def call(self, inputs, training=None):
        # (bs, caps_in, atom)
        mean, centered = custom_ops.center(inputs, [0])
        norm_dim = cfg.training.batch_size
        centered = tf.transpose(centered, [1, 0, 2])
        # centered = tf.reshape(centered, [-1, self.atom])
        mean = K.in_train_phase(mean, self.moving_mean)
        get_inv_sqrt = custom_ops.get_decomposition(self.decomposition, self.iter_num, self.epsilon)

        def train():
            sigma = tf.matmul(centered, centered, transpose_a=True)
            sigma /= (tf.cast(norm_dim, tf.float32) - 1.)
            whiten_matrix = get_inv_sqrt(sigma, self.atom)
            self.add_update([self._assign_moving_average(self.moving_mean, mean, self.momentum, None),
                             self._assign_moving_average(self.moving_matrix, whiten_matrix, self.momentum, None)],
                            inputs)
            return whiten_matrix

        def test():
            moving_matrix = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.atom) * self.epsilon
            return moving_matrix

        inv_sqrt = K.in_train_phase(train, test)
        decorelated = tf.matmul(centered, inv_sqrt)
        decorelated = tf.transpose(decorelated, [1, 0, 2])
        if self.gamma is not None:
            decorelated *= self.gamma
        if self.beta is not None:
            decorelated += self.beta
        # decorelated = tf.reshape(decorelated, [-1, self.caps_in, self.atom])
        return decorelated

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_matrix_initializer': initializers.serialize(self.matrix_initializer)
        }
        base_config = super(CapsDecorelationNormalization4, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsDecorelationNormalization5(Layer):
    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-3,
                 decomposition='iter_norm',
                 iter_num=5,
                 affine=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 moving_mean_initializer='zeros',
                 moving_whiten_initializer='identity',
                 **kwargs):
        assert decomposition in ['zca', 'pca', 'iter_norm']
        super(CapsDecorelationNormalization5, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_whiten_initializer = initializers.get(moving_whiten_initializer)
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.affine = affine
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.atom = 0
        self.caps_out = 0
        self.caps_in = 0

    def matrix_initializer(self, shape, dtype=tf.float32, partition_info=None):
        moving_conv = tf.eye(shape[-1], dtype=dtype)
        moving_conv = tf.reshape(moving_conv, [1, shape[-1], shape[-1]])
        moving_conv = tf.tile(moving_conv, [shape[0], 1, 1])
        return moving_conv

    def build(self, input_shape):
        # shape should be (..., caps_in, caps_out, atom)
        input_shape_list = input_shape.as_list()
        self.atom = input_shape_list[-1]
        self.caps_out = input_shape_list[-2]
        self.caps_in = input_shape_list[-3]
        if self.atom is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.moving_mean = self.add_weight(shape=(1, self.caps_in, 1, self.atom),
                                           name='moving_mean',
                                           synchronization=tf_variables.VariableSynchronization.ON_READ,
                                           initializer=self.moving_mean_initializer,
                                           trainable=False,
                                           aggregation=tf_variables.VariableAggregation.MEAN)
        self.moving_matrix = self.add_weight(shape=(self.caps_in, self.atom, self.atom),
                                             name='moving_matrix',
                                             synchronization=tf_variables.VariableSynchronization.ON_READ,
                                             initializer=self.matrix_initializer,
                                             trainable=False,
                                             aggregation=tf_variables.VariableAggregation.MEAN)

        if self.affine:
            param_shape = (1, self.caps_in, 1, self.atom)
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            self.beta = None

        self.built = True

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def call(self, inputs, training=None):
        # (bs, caps_in, caps_out, atom)
        mean, centered = custom_ops.center(inputs, [0, 2])
        norm_dim = cfg.training.batch_size*self.caps_out
        centered = tf.transpose(centered, [1, 0, 2, 3])  # to [caps_in, caps_out, bs, atom]
        centered = tf.reshape(centered, [self.caps_in, -1, self.atom])
        mean = K.in_train_phase(mean, self.moving_mean)
        get_inv_sqrt = custom_ops.get_decomposition(self.decomposition, self.iter_num, self.epsilon)

        def train():
            sigma = tf.matmul(centered, centered, transpose_a=True)
            sigma /= (tf.cast(norm_dim, tf.float32) - 1.)
            whiten_matrix = get_inv_sqrt(sigma, self.atom)
            self.add_update([self._assign_moving_average(self.moving_mean, mean, self.momentum, None),
                             self._assign_moving_average(self.moving_matrix, whiten_matrix, self.momentum, None)],
                            inputs)
            return whiten_matrix

        def test():
            moving_matrix = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.atom) * self.epsilon
            return moving_matrix

        inv_sqrt = K.in_train_phase(train, test)
        decorelated = tf.matmul(centered, inv_sqrt)
        decorelated = tf.reshape(decorelated, [self.caps_in, -1, self.caps_out, self.atom])
        decorelated = tf.transpose(decorelated, [1, 0, 2, 3])  # to (bs, caps_in, caps_out, atom)

        if self.gamma is not None:
            decorelated *= self.gamma
        if self.beta is not None:
            decorelated += self.beta
        return decorelated

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_matrix_initializer': initializers.serialize(self.matrix_initializer)
        }
        base_config = super(CapsDecorelationNormalization5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VectorNorm(Layer):
    def __init__(self, **kwargs):
        super(VectorNorm, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = custom_ops.vector_norm(inputs, -1)
        return x


def get_original_capsule_layer(inputs, num_out, out_atom, num_routing=3, temper=1.0):
    transformed = CapsuleTransformDense(num_out=num_out, out_atom=out_atom, share_weights=False)(inputs)
    routed = DynamicRouting(num_routing=num_routing, temper=temper)(transformed)
    return routed


def get_factorization_machines(inputs, axis=-2, norm='constant',
                               importance=False, attention=False,
                               temper=1.0, regularize=True):
    # [bs, caps_in, caps_out, atom]
    if regularize:
        cap_in = inputs.get_shape()[1]
        inputs /= np.sqrt(cap_in)
    x1 = tf.reduce_sum(inputs, axis, keepdims=True)  # [bs, 1, caps_out, atom]
    x1a_m, x1a_var = tf.nn.moments(x1, axes=[0,1,2,3])
    x1 = tf.square(x1)
    x1s_m, x1s_var = tf.nn.moments(x1, axes=[0, 1, 2, 3])
    x2_square = tf.square(inputs)
    x2s_m, x2s_var = tf.nn.moments(x2_square, axes=[0, 1, 2, 3])
    x2 = tf.reduce_sum(x2_square, axis, keepdims=True)  # [bs, 1, caps_out, atom]
    x2a_m, x2a_var = tf.nn.moments(x2, axes=[0, 1, 2, 3])
    outputs = x1 - x2
    out_m, out_var = tf.nn.moments(outputs, axes=[0, 1, 2, 3])
    outputs = tf.squeeze(outputs, axis)
    weight = None
    if importance:
        weight = inputs * x1 - x2_square
        weight = tf.reduce_sum(weight, -1, keepdims=True)  # [bs, caps_in, caps_out, 1]
        weight = tf.math.softmax(temper*weight, axis=axis)  # [bs, caps_in, caps_out, 1]
        if attention:
            outputs = outputs * weight
    n = inputs.get_shape().as_list()[axis]
    if norm == 'constant1':
        outputs /= np.sqrt((n-1)*n*2)
    elif norm == 'constant2':
        outputs /= ((n - 1) * n)
    elif norm == 'constant3':
        outputs /= (n - 1)
    elif norm == 'constant4':
        outputs /= ((n - 1)*2)
    elif norm == 'norm':
        mean, var = tf.nn.moments(outputs, -1, keepdims=True)
        outputs = (outputs-mean)/tf.sqrt(var)
    elif norm == 'relu':
        outputs = tf.nn.relu(outputs)
    elif norm == 'cons_relu':
        outputs /= np.sqrt((n - 1) * n * 2)
        outputs = tf.nn.relu(outputs)
    return outputs, weight


def get_average_pooling(inputs):
    x = tf.reduce_mean(inputs, axis=-2)
    return


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


def test_routing_pool():
    x = tf.random.normal([64, 16, 16, 64])
    x = RoutingPooling(3, 2, 8)(x)
    print(x.shape)


def test_batch_2d():
    x = tf.random.normal([128, 32, 32, 3])
    x = batch_2d(x, 2, 2)
    print(x.shape)


def verify_factorization_machines():
    x = tf.random.normal([1000, 16])
    t1 = time.time()
    out1 = get_cross_mul(x)
    t2 = time.time()
    print('cross mul cost:', t2 - t1)
    print('cross mul result:', out1)
    out2 = get_factorization_machines(x)
    t3 = time.time()
    print('FM cost:', t3 - t2)
    print('FM result:', out2)


def test_ablation_fm():
    num = 1000
    atom = 16
    a = tf.random.normal([10000, num, atom], 2, 20)
    a = a / tf.norm(a, axis=-1, keepdims=True)
    a = a / tf.sqrt(tf.cast(num, tf.float32))
    b1 = tf.square(tf.reduce_sum(a, 1))
    # b1_mean, b1_var = tf.nn.moments(b1, 0)
    # print('b1_mean:', b1_mean.numpy())
    # print('b1_var:', b1_var.numpy())
    b2 = tf.reduce_sum(tf.square(a), 1)
    # b2_mean, b2_var = tf.nn.moments(b2, 0)
    # print('b2_mean:', b2_mean.numpy())
    # print('b2_var:', b2_var.numpy())
    fm = b1 - b2
    b3_mean, b3_var = tf.nn.moments(fm, 0)
    print('b3_mean:', b3_mean.numpy())
    print('b3_var:', b3_var.numpy())
    act = tf.reduce_sum(fm, 1)
    act_mean, act_var = tf.nn.moments(act, 0)
    print('act_mean:', act_mean.numpy())
    print('act_var:', act_var.numpy())


def verify_vec_norm():
    vec_norm = VectorNorm()
    x = tf.random.normal([64, 16])
    x = vec_norm(x)
    print(tf.norm(x, axis=-1))


def verify_pri_capsule():
    x = tf.random.normal([10, 8, 8, 64])
    x = CapsuleGroups(height=8,
                      width=8,
                      channel=64,
                      atoms=32,
                      activation='norm')(x)
    print(tf.norm(x, axis=-1))


def verify_dynamic_routing():
    x = tf.random.normal([10, 2, 64, 1, 32])
    y = DynamicRouting()(x)
    print(y)


def verify_transform():
    x = tf.random.normal((128, 30, 8))
    trans = CapsuleTransformDense(10, 16)(x)
    print(trans)


def var_experiments():
    x = tf.random.normal([10240, 64])
    x1_mean, x1_var = tf.nn.moments(x, [0, 1])
    x2_mean, x2_var = tf.nn.moments(x*x, [0, 1])
    x4_mean = tf.reduce_mean(x*x*x*x)
    print('x1_var', x1_var)
    print('x2_var', x2_var)
    print('x4_mean', x4_mean)
    y = get_factorization_machines(x, 1, 'norm')
    y_mean, y_var = tf.nn.moments(y, [0])
    print('y_mean', y_mean)
    print('y_var', y_var)


def var_vec_norm_scale():
    x = tf.random.normal([12800, 160])
    x_norm, _ = custom_ops.vector_norm_scale(x, -1)
    x_norm_verify = tf.norm(x_norm, axis=-1)
    y_mean, y_var = tf.nn.moments(x_norm, [0, 1])
    print('E:',y_mean.numpy(), 'D:', y_var.numpy())


if __name__ == "__main__":
    var_vec_norm_scale()
