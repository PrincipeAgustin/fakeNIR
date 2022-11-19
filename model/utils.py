import tensorflow as tf

IMG_HEIGHT = 960
IMG_WIDTH =  1280

def load(image_file_1, image_file_2):
    # Read and decode an image file to a uint8 tensor
    input_image = tf.io.read_file(image_file_1)
    input_image = tf.io.decode_png(input_image)
    input_image = tf.cast(input_image, tf.float32)

    real_image = tf.io.read_file(image_file_2)
    real_image = tf.io.decode_png(real_image)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def resize(input_image, real_image, height, width):

    input_image = tf.image.resize(input_image, [height, width])
    real_image = tf.image.resize(real_image, [height, width])

    return input_image, real_image

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def de_normalize(image):
    image = (image + 1) * 127.5
    return tf.cast(image, tf.uint8)

def random_jitter(input_image, real_image, img_height = IMG_HEIGHT, img_width = IMG_WIDTH):

    # Transforma la imagen de (h, w, 1) a (h, w, 3)
    real_image = tf.image.grayscale_to_rgb(real_image)

    input_image, real_image = resize(input_image, real_image, int(img_height * 1.117), int(img_width * 1.117))

    stacked_img = tf.stack([input_image, real_image], axis = 0)
    cropped_img = tf.image.random_crop(stacked_img, size=[2, img_height, img_width, 3])

    input_image = cropped_img[0]
    real_image = cropped_img[1]

    if (tf.random.uniform(()) > 0.5):

      input_image = tf.image.flip_left_right(input_image)
      real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image[:,:,0:1]

def load_image_train(image_file, in_path = '.', out_path = '.'):
    input_image, real_image = load(in_path + '/' + image_file, out_path + '/' + image_file)
    input_image, real_image = random_jitter(input_image, real_image)

    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file, in_path = '.', out_path = '.'):
    input_image, real_image = load(in_path + '/' + image_file, out_path + '/' + image_file)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image