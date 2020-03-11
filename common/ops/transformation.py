import tensorflow as tf
import time


def matrix_capsule_matmul(inputs, matrix, num_out):
    # [..., in_num, a, b] X [in_num, out_num, b, c]
    inputs = tf.expand_dims(inputs, -3) # -> [..., in_num, 1, a, b]
    in_shape = inputs.get_shape().as_list()
    in_shape[0] = -1
    # -> [..., in_num, num_out, a, b]
    inputs = tf.tile(inputs, [1 for _ in range(len(in_shape) - 3)] + [num_out] + [1, 1])
    # -> [..., in_num, out_num, b, c]
    matrix = tf.reshape(matrix, [1 for _ in range(len(in_shape) - 4)] + matrix.get_shape().as_list())
    # -> [..., in_num, out_num, a, c]
    result = tf.matmul(inputs, matrix)
    return result


def matrix_capsule_element_wise(inputs, matrix, num_out):
    # [..., in_num, a, b] X [in_num, out_num, b, c]
    in_shape = inputs.get_shape().as_list()
    in_shape[0] = -1
    inputs = tf.expand_dims(inputs, -1)  # -> [..., in_num, a, b, 1]
    inputs = tf.expand_dims(inputs, -4)  # -> [..., in_num, 1, a, b, 1]
    matrix_shape = matrix.get_shape().as_list()
    # -> [..., in_num, num_out, a, b, c]
    inputs = tf.tile(inputs, [1 for _ in range(len(in_shape) - 2)] + [num_out] + [1, 1, matrix_shape[-1]])
    # -> [..., in_num, out_num, 1, b, c]
    matrix = tf.expand_dims(matrix, -3)
    # -> [..., in_num, out_num, a, b, c]
    matrix = tf.tile(matrix, [1 for _ in range(len(matrix_shape)-2)] + [in_shape[-2]] + [1, 1])
    # -> [..., in_num, out_num, a, b, c]
    result = inputs * matrix
    # -> [..., in_num, out_num, a, c]
    result = tf.reduce_sum(result, -2)
    return result


def test_matrix_matmul():
    x = tf.random.normal([2048, 64, 6, 4])
    y = tf.random.normal([64, 32, 4, 7])
    t1 = time.time()
    res1 = matrix_capsule_matmul(x, y, 32)
    t2 = time.time()
    res2 = matrix_capsule_element_wise(x, y, 32)
    t3 = time.time()
    dis = res1 - res2
    print('t1:', t2 - t1)
    print('t2:', t3 - t2)
    print('result:', tf.reduce_sum(dis))


def matmul_element_wise(inputs, matrix, num_out, num_out_dim):
    # inputs shape should be [batch, num_in, num_in_dim] or [batch, height, width, num_in, num_in_dim]
    # matrix shape should be [num_in, num_in_dim, num_out*num_out_dim]
    inputs = tf.expand_dims(inputs, axis=-1)
    # to [batch, num_in, num_in_dim, 1]
    inputs_shape = inputs.get_shape().as_list()
    inputs = tf.tile(inputs, multiples=[1 for _ in range(len(inputs_shape) - 1)] + [num_out * num_out_dim])
    transformed = inputs * matrix
    transformed = tf.reduce_sum(transformed, axis=-2)
    # to [batch, num_in, num_out*num_out_dim]
    transformed = tf.reshape(transformed, [-1] + inputs_shape[1:-2] + [num_out, num_out_dim])
    return transformed


def matmul(inputs, matrix, num_out, num_out_dim):
    # inputs shape should be [batch, num_in, num_in_dim]
    # matrix shape should be [num_in, num_in_dim, num_out*num_out_dim]
    inputs_shape = inputs.get_shape().as_list()
    inputs = tf.expand_dims(inputs, axis=-2)
    # to [batch, num_in, 1, num_in_dim]
    matrix = tf.expand_dims(matrix, 0)
    matrix = tf.tile(matrix, [inputs_shape[0], 1, 1, 1])
    # to [batch, num_in, num_in_dim, num_out*num_out_dim]
    transformed = tf.matmul(inputs, matrix)
    # to [batch, num_in, 1, num_out*num_out_dim]
    transformed = tf.reshape(transformed, inputs_shape[:-1] + [num_out, num_out_dim])
    return transformed


def transform_vec(inputs, num_out, num_out_dim, mode='element_wise'):
    # inputs shape should be [batch, num_in, num_in_dim]
    inputs_shape = inputs.get_shape().as_list()
    num_in = inputs_shape[-2]
    num_in_dim = inputs_shape[-1]
    matrix = tf.Variable(name='matrix',
                         initial_value=tf.random.normal(shape=[num_in, num_out, num_in_dim, num_out_dim], stddev=0.1, dtype=tf.float32),
                         dtype=tf.float32)
    if mode == 'element_wise':
        transformed = matmul_element_wise(inputs, matrix, num_out, num_out_dim)
    else:
        transformed = matmul(inputs, matrix, num_out, num_out_dim)

    return transformed


def transform_spatial_share(inputs, num_out, num_out_dim, mode='element_wise', reuse=False, name=None):
    # inputs shape should be [batch, height, width, channel, num_in_dim]
    inputs_shape = inputs.get_shape().as_list()
    height = inputs_shape[1]
    width = inputs_shape[2]
    channel=inputs_shape[-2]
    num_in_dim=inputs_shape[-1]
    matrix = tf.Variable(name='matrix',
                         initial_value=tf.random.normal(shape=[1, 1, channel, num_out, num_in_dim, num_out_dim], stddev=0.1, dtype=tf.float32),
                         dtype=tf.float32)
    matrix = tf.tile(matrix, multiples=[height, width, 1, 1, 1, 1])
    if mode == 'element_wise':
        transformed = matmul_element_wise(inputs, matrix, num_out, num_out_dim)
    else:
        transformed = matmul(inputs, matrix, num_out)
    return transformed


def test_compare_transform():
    # tf.enable_eager_execution()
    batch_size = 256
    num_in = 128
    num_in_dim = 8
    num_out = 10
    num_out_dim = 16
    inputs = tf.random.normal(shape=[1, num_in, num_in_dim])
    inputs = tf.tile(inputs, [batch_size, 1, 1])

    matrix = tf.Variable(name='matrix',
                         initial_value=tf.random.normal(shape=[num_in, num_in_dim, num_out * num_out_dim], stddev=0.1,
                                                        dtype=tf.float32),
                         dtype=tf.float32)

    t1 = time.time()
    transformed1 = matmul_element_wise(inputs, matrix, num_out, num_out_dim)
    t2 = time.time()
    print('element wise cost:', t2 - t1)
    transformed2 = matmul(inputs, matrix, num_out, num_out_dim)
    t3 = time.time()
    print('matmul cost:', t3 - t2)
    # print('trans1', tf.reduce_sum(transformed1, axis=[1, 2, 3]))
    # print('trans2', tf.reduce_sum(transformed2, axis=[1, 2, 3]))
    print('final result', tf.reduce_sum(transformed2 - transformed1))


def test_tf_matmul():
    batch_size = 128
    matrix = tf.Variable(name='matrix',
                         initial_value=tf.random.normal(shape=[1, 800, 64, 128], stddev=0.1,
                                                        dtype=tf.float32),
                         dtype=tf.float32)
    @tf.function
    def matmul(matrix):
        inputs = tf.random.normal(shape=[1, 800, 1, 64])
        inputs = tf.tile(inputs, [batch_size, 1, 1, 1])

        matrix = tf.tile(matrix, multiples=[batch_size, 1, 1, 1])
        result = tf.matmul(tf.reshape(inputs, [-1, 1, 64]), tf.reshape(matrix, [-1, 64, 128]))
        return result
    result = tf.reshape(matmul(matrix), [-1, 800, 1, 128])
    print(tf.reduce_sum(result, axis=[1, 2, 3]))


def test_tf_matmul2():
    batch_size = 256
    inputs = tf.random.normal(shape=[1, 1, 6400])
    inputs = tf.tile(inputs, [batch_size, 1, 1])
    matrix = tf.Variable(name='matrix',
                         initial_value=tf.random.normal(shape=[1, 6400, 1280], stddev=0.1,
                                                        dtype=tf.float32),
                         dtype=tf.float32)
    matrix = tf.tile(matrix, multiples=[batch_size, 1, 1])
    result = tf.matmul(inputs, matrix)
    print(tf.reduce_sum(result, axis=[1, 2]))


def test_tf_matmul3():
    batch_size = 100000
    inputs = tf.random.normal(shape=[1, 64])
    inputs = tf.tile(inputs, [batch_size, 1])
    matrix = tf.Variable(name='matrix',
                         initial_value=tf.random.normal(shape=[64, 128], stddev=0.1,
                                                        dtype=tf.float32),
                         dtype=tf.float32)
    result = tf.matmul(inputs, matrix)
    print(tf.reduce_sum(result, axis=[1]))


def test_whiten():
    batch_size = 3
    inputs = tf.random.uniform([batch_size, 64, 16], 0, 1)
    fm1 = tf.square(tf.reduce_sum(inputs, [1])) - tf.reduce_sum(tf.square(inputs), [1])
    fm1 = tf.reduce_sum(fm1, -1)
    print(fm1)
    sigma = tf.matmul(inputs, inputs, transpose_b=True)
    s, u, v = tf.linalg.svd(sigma)
    s = tf.linalg.diag(tf.pow(s, -0.5))
    whiten = tf.matmul(tf.matmul(u, s), u, transpose_b=True)
    res = tf.linalg.matrix_transpose(tf.matmul(tf.linalg.matrix_transpose(inputs), whiten))
    fm2 = tf.square(tf.reduce_sum(res, [1])) - tf.reduce_sum(tf.square(res), [1])
    fm2 = tf.reduce_sum(fm2, -1)
    print(fm2)


if __name__ == "__main__":
    test_tf_matmul2()
