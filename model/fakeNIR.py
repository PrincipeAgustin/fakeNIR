import tensorflow as tf
import matplotlib.pyplot as plt
import os
from time import time

from fakeNIRLayers import downsample, upsample
from utils import de_normalize

IMG_HEIGHT = 960
IMG_WIDTH =  1280

BATCH_SIZE = 4
OUTPUT_CHANNELS = 1 # Solo necesitamos un canal de salida.
LAMBDA = 100 # Valor de ajuste de la funcion de perdida

class fakeNIR: 

    def __init__(self, width = IMG_WIDTH, height = IMG_HEIGHT, batch_size = BATCH_SIZE, output_chanels = OUTPUT_CHANNELS, lmd = LAMBDA, work_path = '.', checkpoint_folder = '/checkpoints'):
        
        self.WIDTH = width
        self.HEIHT = height
        self.BATCH_SIZE = batch_size
        self.OUTPUT_CHANNELS = output_chanels
        self.LAMBDA = lmd

        self.work_path = work_path
        # Preveinimos igreso equivocado de paths
        self.CHECK_PATH = os.path.join(work_path, checkpoint_folder)
        self.__check_prefix__ = os.path.join(self.CHECK_PATH, "ckpt")

        self.generator = self.__generator__()
        self.discriminator = self.__discriminator__()

        self.__loss_obj__ = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.__generator_optimizer__ = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.__discriminator_optimizer__ = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.__checkpoint__ = tf.train.Checkpoint(
                                                    generator = self.generator,
                                                    generator_optimizer = self.__generator_optimizer__,
                                                    discriminator = self.discriminator,
                                                    discriminator_optimizer = self.__discriminator_optimizer__,
                                                 )

        try:
            self.__checkpoint__.restore(tf.train.latest_checkpoint(self.CHECK_PATH))
        except:
            pass

    def __generator__(self):

        # Cuidado con que se mete, esta preparado para imagenes de 4 / 3 en caso de
        # querer usar una 16 / 6 habria que ajustar el modelo para que la salida 
        # tenga la misma resolucion 
        inputs = tf.keras.layers.Input(shape=[self.HEIHT, self.WIDTH, 3])               # (batch_size, 960, 1280, 3)

        down_stack = [
            downsample(64, 8, apply_batchnorm=False, strides = (3,4)),      # (batch_size, 320, 320, 64)
            downsample(128, 10, strides=5),                                 # (batch_size, 64, 64, 128)
            downsample(256, 4),                                             # (batch_size, 32, 32, 256)
            downsample(512, 4),                                             # (batch_size, 16, 16, 512)
            downsample(512, 4),                                             # (batch_size, 8, 8, 512)
            downsample(512, 4),                                             # (batch_size, 4, 4, 512)
            downsample(512, 4),                                             # (batch_size, 2, 2, 512)
            downsample(512, 4),                                             # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),                           # (batch_size, 2, 2, 512)
            upsample(512, 4, apply_dropout=True),                           # (batch_size, 4, 4, 512)
            upsample(512, 4, apply_dropout=True),                           # (batch_size, 8, 8, 512)
            upsample(512, 4),                                               # (batch_size, 16, 16, 512)
            upsample(256, 4),                                               # (batch_size, 32, 32, 256)
            upsample(128, 4),                                               # (batch_size, 64, 64, 128)
            upsample(64, 10, strides=5),                                    # (batch_size, 320, 320, 64)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 8,
                                                strides=(3, 4),
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh')          # (batch_size, 960, 1280, 1)

        x = inputs

        # Aplicamos el encoder y cargamos los skips conections
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Aplicamos el encoder y el skip conection de cada capa del encoder
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def __discriminator__(self):

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.HEIHT, self.WIDTH, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.HEIHT, self.WIDTH, 1], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])                                     # (batch_size, 960, 1280, 4 (3 + 1))

        down1 = downsample(64, 8, apply_batchnorm=False, strides = (3,4))(x)        # (batch_size, 320, 320, 64)
        down2 = downsample(128, 10, strides = 5)(down1)                                  # (batch_size, 64, 64, 128)
        down3 = downsample(256, 4)(down2)                                               # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)                              # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)                      # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)                         # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)      # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
    
        # Diferencia entre todos true por ser real y lo que detecta el discriminador
        real_loss = self.__loss_obj__(tf.ones_like(disc_real_output), disc_real_output)

        # Diferencia entre todos false por ser generado y lo que detecta el discriminador
        generated_loss = self.__loss_obj__(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):

        # Evalua diferencia entre un true por imagen real y un true 
        gan_loss = self.__loss_obj__(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error 
        # Cambiar 
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def predict(self, image):

        return self.generator(image, training=False)

    def generate_images(self, test_input, tar, save_filename = '', show_img = False):

        prediction = self.generator(test_input, training=False)

        display_list = [test_input[0], tar[0][..., 0], prediction[0][..., 0], tar[0][..., 0] - prediction[0][..., 0]]
        title = ['Input Image', 'NIR Truth', 'Predicted Image', 'Pix 2 Pix diference']
        cmap = [None, None, None, 'RdYlGn']

        if save_filename != '': 
            tf.keras.preprocessing.image.save_img(save_filename + '.jpg', prediction[0])

        if show_img:
            plt.figure(figsize=(30,15));
            for i in range(4):
                plt.subplot(1, 4, i+1);
                plt.title(title[i]);
                plt.imshow(de_normalize(display_list[i]), cmap=cmap[i]);
                plt.axis('off');
            plt.show();

    @tf.function
    def __traing_step__(self, input_im, target_im):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            gen_output = self.generator(input_im, training=True)

            disc_real_output = self.discriminator([input_im, target_im], training=True)
            disc_generated_output = self.discriminator([input_im, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target_im)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.__generator_optimizer__.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.__discriminator_optimizer__.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

            return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss 

    def train(self, dataset: tf.data.Dataset, epochs, epochs_to_save = 20, size = 0, start_epoch = 0, test_dataset: tf.data.Dataset = None, test_images = 0, save_imgs_path = '', show_images = False):

        if not size:

            size = dataset.cardinality().numpy()

        for epoch in range(start_epoch, epochs):

            start = time()
            img_segs = start

            img = 0
            img_path = save_imgs_path

            for input_im, target_im in dataset.take(size):

                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.__traing_step__(input_im, target_im)
                img_segs = time()
                print('Epoch: {} / {} - Train: {} / {} - Total time: {:.2f} s - Segs per step: {:.2f}\ngen_total_loss: {} - gen_gan_loss: {} - gen_l1_loss: {} - disc_loss: {}'
                        .format(
                            epoch, 
                            epochs, 
                            img, 
                            size, 
                            time() - start, 
                            time() - img_segs,
                            gen_total_loss,
                            gen_gan_loss,
                            gen_l1_loss,
                            disc_loss
                        )
                    )
                # Leave last
                img+=1

            if test_dataset != None:

                img = 0
                for imp, tar in test_dataset.take(test_images):

                    if save_imgs_path != '':

                        img_path += ('/' + str(img) + '' + str(epoch + 1)).replace('//', '/')
                    
                    self.generate_images(imp, tar, img_path, save_imgs_path)
                    img+=1

            
            if (epoch + 1) % epochs_to_save == 0:

                self.__checkpoint__.save(self.__check_prefix__())

