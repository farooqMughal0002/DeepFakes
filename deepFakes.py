from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate, add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def inception_resnet_v2(input_shape, num_classes=2):
    img_input = Input(shape=input_shape)
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = MaxPooling2D(3, strides=2)(x)

    # 10 x Inception-ResNet-A block
    for i in range(10):
        x = inception_resnet_a_block(x, scale=0.17)

    # Reduction-A block: 17 x 17 x 1088
    x = reduction_a_block(x)

    # 20 x Inception-ResNet-B block
    for i in range(20):
        x = inception_resnet_b_block(x, scale=0.1)

    # Reduction-B block: 8 x 8 x 2080
    x = reduction_b_block(x)

    # 10 x Inception-ResNet-C block
    for i in range(10):
        x = inception_resnet_c_block(x, scale=0.2)

    # Final convolution block: 1 x 1 x 1536
    x = conv2d_bn(x, 1536, 1, name='final_conv')

    # Classification block
    x = AveragePooling2D(8, name='avg_pool')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='inception_resnet_v2')

    return model
