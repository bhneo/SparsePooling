from tensorflow.python.keras import layers, regularizers, initializers

WEIGHT_DECAY = 1e-4
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997


def bn_relu(inputs, axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, bn_name=None):
    x = layers.BatchNormalization(axis=axis,
                                  momentum=momentum,
                                  epsilon=epsilon,
                                  name=bn_name)(inputs)
    x = layers.ReLU()(x)
    return x


def conv_bn_relu(inputs, filters=64, kernel_size=(7, 7), strides=(2, 2),
                 use_bias=True,
                 kernel_initializer=initializers.he_normal(),
                 kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                 bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                 name=None):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding='same', use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name='res'+name)(inputs)
    return bn_relu(x, axis=bn_axis, momentum=momentum, epsilon=epsilon, bn_name='bn'+name)


def conv_bn(inputs, filters=64, kernel_size=(3, 3), strides=(1, 1),
            use_bias=True,
            kernel_initializer=initializers.he_normal(),
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
            name=None):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding='same', use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name='res'+name)(inputs)
    return layers.BatchNormalization(axis=bn_axis, momentum=momentum, epsilon=epsilon, name='bn'+name)(x)


def bn_relu_conv(inputs, filters, kernel_size, strides=(1, 1), padding='same',
                 use_bias=True,
                 kernel_initializer=initializers.he_normal(),
                 kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                 bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                 name=None):
    x = bn_relu(inputs, axis=bn_axis, momentum=momentum, epsilon=epsilon, bn_name='bn'+name)
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding, use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name='res'+name)(x)
    return x


def shortcut_v1(inputs, residual,
                use_bias=True,
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                name=None):
    input_shape = inputs.get_shape().as_list()
    residual_shape = residual.get_shape().as_list()
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    equal_channels = input_shape[3] == residual_shape[3]

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        inputs = layers.Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                               strides=(stride_width, stride_height), use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=kernel_regularizer,
                               name='res'+name)(inputs)
        inputs = layers.BatchNormalization(axis=bn_axis, momentum=momentum, epsilon=epsilon, name='bn'+name)(inputs)
    return layers.add([inputs, residual])


def shortcut_v2(inputs, residual,
                use_bias=True,
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                name=None):
    input_shape = inputs.get_shape().as_list()
    residual_shape = residual.get_shape().as_list()
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    equal_channels = input_shape[3] == residual_shape[3]

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        inputs = layers.Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                               strides=(stride_width, stride_height), use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=kernel_regularizer,
                               name='res'+name)(inputs)
    return layers.add([inputs, residual])


def basic_block_v1(inputs, filters,
                   stage, block, use_bias=True,
                   init_strides=(1, 1), is_first_block_of_first_layer=False,
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                   bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON):
    if isinstance(filters, int):
        filters1, filters2 = filters, filters
    else:
        filters1, filters2 = filters
    base_name = str(stage) + block + '_branch'
    x = conv_bn_relu(inputs, filters=filters1, kernel_size=(3, 3), strides=init_strides,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                     name=base_name+'2a')
    x = conv_bn(x, filters=filters2, kernel_size=(3, 3), strides=(1, 1),
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                name=base_name+'2b')
    x = shortcut_v1(inputs, x,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                    name=base_name+'1')
    return layers.ReLU()(x)


def basic_block_v2(inputs, filters,
                   stage, block,
                   use_bias=True,
                   init_strides=(1, 1), is_first_block_of_first_layer=False,
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                   bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON):
    if isinstance(filters, int):
        filters1, filters2 = filters, filters
    else:
        filters1, filters2 = filters
    base_name = str(stage) + block + '_branch'
    if is_first_block_of_first_layer:
        x = layers.Conv2D(filters=filters1, kernel_size=(3, 3), use_bias=use_bias,
                          strides=init_strides, padding='same',
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer,
                          name=base_name+'2a')(inputs)
    else:
        x = bn_relu_conv(inputs, filters=filters1, kernel_size=(3, 3), use_bias=use_bias,
                         strides=init_strides,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                         name=base_name+'2a')

    residual = bn_relu_conv(x, filters=filters2, kernel_size=(3, 3), strides=(1, 1),
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                            name=base_name+'2b')
    return shortcut_v2(inputs, residual,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       name=base_name+'1')


def bottleneck_v1(inputs, filters,
                  stage, block,
                  use_bias=True,
                  init_strides=(1, 1), is_first_block_of_first_layer=False,
                  kernel_initializer=initializers.he_normal(),
                  kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                  bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON):
    if isinstance(filters, int):
        filter1, filter2, filter3 = filters, filters, 4*filters
    else:
        filter1, filter2, filter3 = filters
    base_name = str(stage) + block + '_branch'
    x = conv_bn_relu(inputs, filters=filter1, kernel_size=(1, 1), strides=init_strides, use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                     name=base_name + '2a')
    x = conv_bn_relu(x, filters=filter2, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                     name=base_name + '2b')
    x = conv_bn(x, filters=filter3, kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                name=base_name + '2c')
    x = shortcut_v1(inputs, x,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bn_axis=bn_axis, momentum=momentum, epsilon=epsilon,
                    name=base_name+'1')
    return layers.ReLU()(x)


def bottleneck_v2(inputs, filters, init_strides=(1, 1),
                  use_bias=True,
                  is_first_block_of_first_layer=False,
                  kernel_initializer=initializers.he_normal(),
                  kernel_regularizer=regularizers.l2(WEIGHT_DECAY)):
    # TODO: not verified
    if is_first_block_of_first_layer:
        x = layers.Conv2D(filters=filters, kernel_size=(1, 1),
                          strides=init_strides, use_bias=use_bias, padding='same',
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(inputs)
    else:
        x = bn_relu_conv(inputs, filters=filters, kernel_size=(1, 1),
                         strides=init_strides, use_bias=use_bias)

    x = bn_relu_conv(x, filters=filters, kernel_size=(3, 3), use_bias=use_bias)
    x = bn_relu_conv(x, filters=filters * 4, kernel_size=(1, 1), use_bias=use_bias)
    return shortcut_v2(inputs, x, use_bias=use_bias)


def residual_block(inputs, block_function, filters, repetitions, stage, use_bias=True,
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                   bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON):
    x = inputs
    for i in range(repetitions):
        init_strides = (1, 1)
        if i == 0 and stage != 2:
            init_strides = (2, 2)
        x = block_function(x, filters=filters,
                           stage=stage, block='block_%d' % (i + 1), use_bias=use_bias,
                           init_strides=init_strides,
                           is_first_block_of_first_layer=(stage == 2 and i == 0),
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bn_axis=bn_axis, momentum=momentum, epsilon=epsilon)
    return x


def build_resnet_backbone(inputs, layer_num, repetitions=None, start_filters=16,
                          arch='cifar', use_bias=True,
                          kernel_initializer=initializers.he_normal(),
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                          bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                          version='v1'):
    if arch == 'cifar':
        if repetitions is None:
            if layer_num == 18:
                repetitions = [2, 2, 2, 2]
            elif layer_num == 20:
                repetitions = [3, 3, 3]
            elif layer_num == 32:
                repetitions = [5, 5, 5]
            elif layer_num == 44:
                repetitions = [7, 7, 7]
            elif layer_num == 56:
                repetitions = [9, 9, 9]
            elif layer_num == 110:
                repetitions = [18, 18, 18]
            else:
                repetitions = [3, 3, 3]

        x = layers.Conv2D(filters=start_filters, kernel_size=(3, 3),
                          strides=(1, 1), padding='same', use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer,
                          name='conv1')(inputs)
        x = bn_relu(x, axis=bn_axis, momentum=momentum, epsilon=epsilon, bn_name='bn_conv1')

        block = x
        filters = start_filters
        if version == 'v1':
            block_fn = basic_block_v1
        elif version == 'v2':
            block_fn = basic_block_v2
        for i, r in enumerate(repetitions):
            block = residual_block(block, block_fn, filters=filters, repetitions=r, stage=i+2, use_bias=use_bias,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bn_axis=bn_axis, momentum=momentum, epsilon=epsilon)
            filters *= 2
        if version == 'v2':
            block = bn_relu(block, axis=bn_axis, momentum=momentum, epsilon=epsilon)
        return block
    elif arch == 'imagenet':
        if layer_num == 18:
            repetitions = [2, 2, 2, 2]
        elif layer_num == 34:
            repetitions = [3, 4, 6, 3]
        elif layer_num == 50:
            repetitions = [3, 4, 6, 3]
        elif layer_num == 101:
            repetitions = [3, 4, 23, 3]
        elif layer_num == 152:
            repetitions = [3, 8, 36, 3]
        else:
            repetitions = [2, 2, 2, 2]

        x = layers.Conv2D(filters=start_filters, kernel_size=(7, 7),
                          strides=(2, 2), padding='same', use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer,
                          name='conv1')(inputs)
        x = bn_relu(x, axis=bn_axis, momentum=momentum, epsilon=epsilon, bn_name='bn_conv1')
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        block = x
        filters = start_filters
        if layer_num >= 50:
            block_fn = bottleneck_v1
        else:
            block_fn = basic_block_v1
        for i, r in enumerate(repetitions):
            block = residual_block(block, block_fn, filters=filters, repetitions=r, stage=i + 2, use_bias=use_bias,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bn_axis=bn_axis, momentum=momentum, epsilon=epsilon)
            filters *= 2
        return block

