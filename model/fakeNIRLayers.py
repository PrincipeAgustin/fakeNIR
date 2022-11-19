import tensorflow as tf

def downsample(filters, size, apply_batchnorm=True, strides = 2):

    # Valores iniciales aleatorios
    initializer = tf.keras.initializers.RandomNormal(0.0, 0.02)

    # Creamos la capa
    result = tf.keras.Sequential()

    # Agregamos la capa de convolución
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                                kernel_initializer=initializer, use_bias=(not apply_batchnorm)))

    # En caso de necesitarlo aplicamos batch normalization
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    # Agregamos la capa de activacion al final
    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False, strides = 2):

    # Valores iniciales aleatorios
    initializer = tf.random_normal_initializer(0., 0.02)

    # Creamos la capa
    result = tf.keras.Sequential()

    # Agregamos la capa de de-convolución
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    # Aplicamos batch normalization siempre, por eso no lleva bias
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result