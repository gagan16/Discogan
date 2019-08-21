from __future__ import print_function, division
import imageio
import tensorflow as tf

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
from glob import glob

class DiscoGAN():

    def __init__(self, sess, args):
        self.model_name = "Discogan"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir

        self.log_dir = args.log_dir
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.image_size = args.img_size
        self.learning_rate = args.learning_rate
        self.print_freq = args.print_freq
        self.c_dim = 1
        self.channel = 3
        self.z_dim = 128
        self.image_shape = [self.image_size, self.image_size, self.channel]

        print()

        print("##### Information #####")
        print("# GAN:", self.model_name)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)

        print("# Image size : ", self.image_size)
        print("# learning rate : ", self.learning_rate)

        print()



    def build_model(self):
        # Configure data loader

        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.image_size, self.image_size))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.image_size / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        #optimizer = Adam(0.0002, 0.5)
        optimizer = Adam(lr=0.00025, beta_1=  0.5, beta_2=  0.999)

        # Build and compile the discriminators
        print("##Discriminitor structure##")
        self.d_A = self.build_discriminator()

        # self.d_A.summary()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        print("##Generator structure##")
        self.g_AB.summary()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.image_shape)
        img_B = Input(shape=self.image_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)



        # Objectives
        # + Adversarial: Fool domain discriminators
        # + Translation: Minimize MAE between e.g. fake B and true B
        # + Cycle-consistency: Minimize MAE between reconstructed images and original
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        fake_B, fake_A,
                                        reconstr_A, reconstr_B ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, normalize=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalize:
                d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.image_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, normalize=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channel, kernel_size=4, strides=1,
                            padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.image_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self):
        self.build_model()
        start_time = datetime.datetime.now()

        path = glob('dataset/try/train/*' )
        number_of_batches = int(len(path) / self.batch_size)

        tf.initialize_all_variables().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # saving checkpoints
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / number_of_batches)
            start_batch_id = checkpoint_counter - start_epoch * number_of_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS ", counter)
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")


        # Adversarial loss ground truths
        valid = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        for epoch in range(start_epoch,self.epoch):

            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(self.batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                losses = np.empty(shape=1)
                losses = np.append(losses, dA_loss)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                losses = np.append(losses, dB_loss)


                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, \
                                                                         imgs_B, imgs_A, \
                                                                         imgs_A, imgs_B])

                losses = np.append(losses, g_loss)

                self.write_to_tensorboard(batch_i, self.writer, losses)

                elapsed_time = datetime.datetime.now() - start_time
                counter += 1
                # Plot the progress
                print ("[%d] [%d/%d] time: %s, [d_loss: %f, g_loss: %f]" % (epoch, batch_i,
                                                                        self.data_loader.n_batches,
                                                                        elapsed_time,
                                                                        d_loss[0], g_loss[0]))

                # If at save interval => save generated image samples
                if batch_i % self.print_freq == 0:
                    self.sample_images(epoch, batch_i)

                start_batch_id = 0
                # print(counter)
                self.save(self.checkpoint_dir, counter)
                # self.visualize_results(epoch)
            print("main counter", counter)
            self.save(self.checkpoint_dir, counter)

#     def sample_images(self, epoch, batch_i):
#
#         os.makedirs(self.result_dir+'/' +self.dataset_name, exist_ok=True)
#
#         imgs_A, imgs_B = self.data_loader.load_data(batch_size=1, is_testing=True)
#
#         # Translate images to the other domain
#         fake_B = self.g_AB.predict(imgs_A)
# #        fake_A = self.g_BA.predict(imgs_B)
#
# #        # Rescale images 0 - 1
#         fake_B = 0.5 * fake_B + 0.5
#
#         # imageio.imwrite('filename.jpg', array)
#         imageio.imwrite(self.result_dir+'/' +self.dataset_name+'/%d_Fake%d.png' % (self.dataset_name, epoch, batch_i), fake_B[0])
    def sample_images(self, epoch, batch_i):
        os.makedirs(self.result_dir+'/' +self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=1, is_testing=True)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(self.result_dir+'/' +self.dataset_name+'/%d_%d.png' % (epoch, batch_i))
        plt.close()
        os.makedirs(self.result_dir+'/' +self.dataset_name+'/single', exist_ok=True)
        fake_B = 0.5 * fake_B + 0.5
        imageio.imwrite(self.result_dir+'/' +self.dataset_name+'/single/%d_Fake%d.png' % ( epoch, batch_i), fake_B[0])

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def write_to_tensorboard(self, generator_step, summary_writer,
                             losses):

        summary = tf.Summary()

        value = summary.value.add()
        value.simple_value = losses[1]
        value.tag = 'Critic Real Loss'

        value = summary.value.add()
        value.simple_value = losses[2]
        value.tag = 'Critic Fake Loss'

        value = summary.value.add()
        value.simple_value = losses[3]
        value.tag = 'Generator Loss'

        value = summary.value.add()
        value.simple_value = losses[1] - losses[2]
        value.tag = 'Critic Loss (D_real - D_fake)'

        value = summary.value.add()
        value.simple_value = losses[1] + losses[2]
        value.tag = 'Critic Loss (D_fake + D_real)'

        summary_writer.add_summary(summary, generator_step)
        summary_writer.flush()


