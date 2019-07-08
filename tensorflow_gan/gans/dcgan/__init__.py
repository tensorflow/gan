from tensorflow.keras import layers  # pylint: disable=import-error
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout  # nopep8 pylint: disable=import-error
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU  # nopep8 pylint: disable=import-error
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, GlobalMaxPooling2D   # nopep8 pylint: disable=import-error
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error


optimizer = Adam(0.0002, beta_1=0.5)


def generator(input_shape, img_shape, filters=1024, num_blocks=4,
              reshape_size=4, alpha=0.02):

    inputs = layers.Input(shape=input_shape)

    dense_dims = reshape_size * reshape_size * filters
    x = Dense(dense_dims)(inputs)
    x = Reshape((reshape_size, reshape_size, filters))(x)

    filters = int(filters/2)
    x = Conv2DTranspose(filters, kernel_size=3, strides=2,
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)

    for _ in range(num_blocks - 1):
        filters = int(filters/2)
        x = Conv2DTranspose(filters, kernel_size=3, strides=1,
                            padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        # x = Activation('relu')(x)

    x = Conv2DTranspose(img_shape[2], kernel_size=3, strides=2,
                        padding='same')(x)
    outputs = Activation('tanh')(x)
    return Model(inputs, outputs)


def discriminator(input_shape, filters=64, num_blocks=4, alpha=0.02):
    inputs = layers.Input(shape=input_shape)

    x = Conv2D(filters, kernel_size=3, padding="valid")(inputs)
    x = LeakyReLU(alpha=alpha)(x)

    for _ in range(num_blocks - 1):
        filters *= 2
        x = Conv2D(filters, kernel_size=3, padding="valid")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)

    x = GlobalMaxPooling2D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)
