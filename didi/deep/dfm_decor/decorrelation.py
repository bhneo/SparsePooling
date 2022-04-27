import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.python.ops import variables as tf_variables


class DecorelationNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 center_axis=0,
                 group_axis=1,
                 momentum=0.99,
                 epsilon=1e-3,
                 m_per_group=0,
                 decomposition='cholesky',
                 iter_num=5,
                 moving_mean_initializer='zeros',
                 moving_cov_initializer='identity',
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 device='cpu',
                 affine=True,
                 **kwargs):
        assert decomposition in ['cholesky', 'zca', 'pca', 'iter_norm',
                                 'cholesky_wm', 'zca_wm', 'pca_wm', 'iter_norm_wm']
        super(DecorelationNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.m_per_group = m_per_group
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        # self.moving_cov_initializer = initializers.get(moving_cov_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.center_axis = center_axis
        self.group_axis = group_axis
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.device = device
        self.affine = affine

    def matrix_initializer(self, shape, dtype=tf.float32, partition_info=None):
        moving_convs = []
        for i in range(shape[0]*shape[1]):
            moving_conv = tf.expand_dims(tf.eye(shape[2], dtype=dtype), 0)
            moving_convs.append(moving_conv)

        moving_convs = tf.concat(moving_convs, 0)
        moving_convs = tf.reshape(moving_convs, shape)
        return moving_convs

    def build(self, input_shape):
        shape = input_shape.as_list()
        dim = shape[self.group_axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if self.m_per_group == 0:
            self.m_per_group = dim
        self.group = dim // self.m_per_group
        assert (dim % self.m_per_group == 0), 'dim is {}, m is {}'.format(dim, self.m_per_group)

        self.moving_mean = None
        self.moving_matrix = None
        if self.center_axis == 0:
            mean_shape = [1] + shape[1:]
            self.moving_mean = self.add_weight(shape=mean_shape,
                                               name='moving_mean',
                                               initializer=self.moving_mean_initializer,
                                               trainable=False,
                                               aggregation=tf_variables.VariableAggregation.MEAN)
            self.moving_matrix = self.add_weight(shape=(shape[1], self.group, self.m_per_group, self.m_per_group),
                                                 name='moving_matrix',
                                                 initializer=self.matrix_initializer,
                                                 trainable=False,
                                                 aggregation=tf_variables.VariableAggregation.MEAN)

        if self.affine:
            param_shape = [1] + shape[1:]
            self.gamma = self.add_weight(name='gamma',
                                         shape=param_shape,
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         trainable=True,
                                         experimental_autocast=False)
            self.beta = self.add_weight(name='beta',
                                        shape=param_shape,
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        trainable=True,
                                        experimental_autocast=False)

        self.built = True

    def call(self, inputs, training=None):
        _, n, k = K.int_shape(inputs)
        bs = K.shape(inputs)[0]

        mean, centered = center(inputs, self.moving_mean, self.center_axis)
        if self.center_axis == 0:
            centered = tf.transpose(centered, (1, 2, 0))
        elif self.center_axis == 1:
            centered = tf.transpose(centered, (0, 2, 1))
        elif self.center_axis == 2:
            pass
        else:
            raise ValueError('Axis invalid!')
        get_inv_sqrt = get_decomposition(self.decomposition,
                                         bs,
                                         self.group,
                                         self.center_axis,
                                         self.iter_num,
                                         self.epsilon,
                                         self.device)

        def train():
            covariance = get_group_cov(centered, self.group, self.m_per_group)

            covariance = (1 - self.epsilon) * covariance + \
                         tf.expand_dims(tf.expand_dims(tf.eye(self.m_per_group) * self.epsilon, 0), 0)

            whiten_matrix = get_inv_sqrt(covariance, self.m_per_group)[1]

            if self.center_axis == 0:
                updates = [K.moving_average_update(self.moving_mean,
                                                   mean,
                                                   self.momentum),
                           K.moving_average_update(self.moving_matrix,
                                                   whiten_matrix if '_wm' in self.decomposition else covariance,
                                                   self.momentum)]

                self.add_update(updates, inputs)

            return whiten_matrix

        def test():
            moving_matrix = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.m_per_group) * self.epsilon
            if '_wm' in self.decomposition:
                return moving_matrix
            else:
                return get_inv_sqrt(moving_matrix, self.m_per_group)[1]

        shape = K.int_shape(centered)
        inv_sqrt = train()
        if self.center_axis == 0:
            centered = tf.reshape(centered, [shape[0], self.group, self.m_per_group, -1])
            whitened = tf.matmul(inv_sqrt, centered)
            whitened = K.reshape(whitened, [shape[0], shape[1], -1])
            whitened = tf.transpose(whitened, [2, 0, 1])
        else:
            centered = tf.reshape(centered, [-1, self.group, self.m_per_group, shape[2]])
            whitened = tf.matmul(inv_sqrt, centered)
            whitened = K.reshape(whitened, [-1, shape[1], shape[2]])
            if self.center_axis == 1:
                whitened = tf.transpose(whitened, [0, 2, 1])

        if self.affine:
            whitened = whitened * self.gamma + self.beta

        return whitened

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_matrix_initializer': initializers.serialize(self.matrix_initializer)
        }
        base_config = super(DecorelationNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def center(inputs, moving_mean, axis=0):
    _, n, k = K.int_shape(inputs)
    mean = tf.reduce_mean(inputs, axis=axis, keepdims=True)
    if axis == 0:
        mean = K.in_train_phase(mean, moving_mean)
    centered = inputs - mean

    return mean, centered


def get_decomposition(decomposition, batch_size, group, axis, iter_num, epsilon, device='cpu'):
    if device == 'cpu':
        device = '/cpu:0'
    else:
        device = '/gpu:0'
    if decomposition == 'cholesky' or decomposition == 'cholesky_wm':
        def get_inv_sqrt(covariance, m_per_group):
            with tf.device(device):
                sqrt = tf.linalg.cholesky(covariance)
            if axis == 0:
                inv_sqrt = tf.linalg.triangular_solve(sqrt,
                                                      tf.tile(tf.expand_dims(tf.eye(m_per_group), 0),
                                                              [group, 1, 1]))
            else:
                inv_sqrt = tf.linalg.triangular_solve(sqrt,
                                                      tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(m_per_group), 0), 0),
                                                              [batch_size, group, 1, 1]))
            return sqrt, inv_sqrt
    elif decomposition == 'zca' or decomposition == 'zca_wm':
        def get_inv_sqrt(covariance, m_per_group):
            with tf.device(device):
                S, U, _ = tf.linalg.svd(covariance + tf.eye(m_per_group) * epsilon, full_matrices=True)
            D = tf.linalg.diag(tf.pow(S, -0.5))
            inv_sqrt = tf.matmul(tf.matmul(U, D), U, transpose_b=True)
            D = tf.linalg.diag(tf.pow(S, 0.5))
            sqrt = tf.matmul(tf.matmul(U, D), U, transpose_b=True)
            return sqrt, inv_sqrt
    elif decomposition == 'pca' or decomposition == 'pca_wm':
        def get_inv_sqrt(covariance, m_per_group):
            with tf.device(device):
                S, U, _ = tf.linalg.svd(covariance + tf.eye(m_per_group) * epsilon, full_matrices=True)
            D = tf.linalg.diag(tf.pow(S, -0.5))
            inv_sqrt = tf.matmul(D, U, transpose_b=True)
            D = tf.linalg.diag(tf.pow(S, 0.5))
            sqrt = tf.matmul(D, U, transpose_b=True)
            return sqrt, inv_sqrt
    elif decomposition == 'iter_norm' or decomposition == 'iter_norm_wm':
        def get_inv_sqrt(covariance, m_per_group):
            trace = tf.linalg.trace(covariance)
            trace = tf.expand_dims(trace, [-1])
            trace = tf.expand_dims(trace, [-1])
            sigma_norm = covariance / trace

            projection = tf.eye(m_per_group)
            projection = tf.expand_dims(projection, 0)
            projection = tf.expand_dims(projection, 0)
            # projection = tf.tile(projection, [group, 1, 1])
            for i in range(iter_num):
                projection = (3 * projection - projection * projection * projection * sigma_norm) / 2

            return None, projection / tf.sqrt(trace)
    else:
        assert False
    return get_inv_sqrt


def get_group_cov(inputs, group, m_per_group):
    # bs, n, k
    shape = list(K.int_shape(inputs))
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = -1
    shape = [shape[0], group, m_per_group, shape[2]]
    grouped = tf.reshape(inputs, shape)
    conv = tf.matmul(grouped, grouped, transpose_b=True)

    conv /= (tf.cast(K.shape(inputs)[-1], tf.float32) - 1.)
    return conv


if __name__ == '__main__':
    a = tf.random.normal(shape=[512, 358, 12])
    center_axis = 0
    group_axis = 2
    m = 12
    decomposition = 'iter_norm'
    iter_num = 5
    out = DecorelationNormalization(center_axis=center_axis,
                                    group_axis=group_axis,
                                    m_per_group=m,
                                    decomposition=decomposition,
                                    iter_num=iter_num)(a)
    print(out.shape)
