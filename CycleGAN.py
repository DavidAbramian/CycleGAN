from keras.layers import Layer, Input, Dropout, Conv2D, Activation, add, BatchNormalization, UpSampling2D, \
    Conv2DTranspose, Flatten
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network

# from collections import OrderedDict
# from scipy.misc import toimage  # has depricated
import matplotlib.image as mpimage
# import cv2
import numpy as np
import datetime
import time
import json
import csv
import sys
import os

import keras.backend as K
import tensorflow as tf

from loadData import load_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# np.random.seed(seed=12345)

class CycleGAN():
    def __init__(self, image_folder='retina'):

        # ======= Data ==========
        print('--- Caching data ---')

        data = load_data(subfolder=image_folder)

        self.channels = data["nr_of_channels"]
        self.img_shape = data["image_size_A"] + (self.channels,)
        
        self.channels_A = data["nr_of_channels_A"]
        self.img_shape_A = data["image_size_A"] + (self.channels_A,)
        
        self.channels_B = data["nr_of_channels_B"]
        self.img_shape_B = data["image_size_B"] + (self.channels_B,)

        print('Image shape: ', self.img_shape)

        self.A_train = data["trainA_images"]
        self.B_train = data["trainB_images"]
        self.A_test = data["testA_images"]
        self.B_test = data["testB_images"]
        self.testA_image_names = data["testA_image_names"]
        self.testB_image_names = data["testB_image_names"]

        self.paired_data = True

        # ===== Model parameters ======
        # Training parameters
        self.lambda_ABA = 10.0  # Cyclic loss weight A_2_B
        self.lambda_BAB = 10.0  # Cyclic loss weight B_2_A
        self.lambda_adversarial = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = 2e-4
        self.learning_rate_G = 2e-4
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.synthetic_pool_size = 50  # Size of image pools used for training the discriminators
        self.beta_1 = 0.5  # Adam parameter
        self.beta_2 = 0.999  # Adam parameter
        self.batch_size = 5  # Number of images per batch
        self.epochs = 200  # choose multiples of 20 since the models are saved each 20th epoch

        self.save_models = False  # Save or not the generator and discriminator models
        self.save_training_img = True  # Save or not example training results or only tmp.png
        self.save_training_img_interval = 1  # Number of epoch between saves of intermediate training results
        self.tmp_img_update_frequency = 3  # Number of batches between updates of tmp.png

        # Architecture parameters
        self.use_instance_normalization = True  # Use instance normalization or batch normalization
        self.use_dropout = False  # Dropout in residual blocks
        self.use_bias = True  # Use bias
        self.use_linear_decay = True  # Linear decay of learning rate, for both discriminators and generators
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start
        self.use_patchgan = True  # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_resize_convolution = False  # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.discriminator_sigmoid = True

        # Tweaks
        self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # ===== Architecture =====
        # Normalization
        if self.use_instance_normalization:
            self.normalization = InstanceNormalization
        else:
            self.normalization = BatchNormalization

        # Optimizers
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # Build discriminators
        D_A = self.build_discriminator()
        D_B = self.build_discriminator()

        # Define discriminator models
        image_A = Input(shape=self.img_shape)
        image_B = Input(shape=self.img_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')

        # Compile discriminator models
        loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)

        # Use containers to make a static copy of discriminators, used when training the generators
        self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')

        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Build generators
        self.G_A2B = self.build_generator(name='G_A2B_model')
        self.G_B2A = self.build_generator(name='G_B2A_model')

        # Define full CycleGAN model, used for training the generators
        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        # Compile full CycleGAN model
        model_outputs = [reconstructed_A, reconstructed_B,
                         dB_guess_synthetic, dA_guess_synthetic]
        compile_losses = [self.cycle_loss, self.cycle_loss,
                          self.lse, self.lse]
        compile_weights = [self.lambda_ABA, self.lambda_BAB,
                           self.lambda_adversarial, self.lambda_adversarial]

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)

        # ===== Folders and configuration =====
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + image_folder

        # Output folder for run data and images
        self.out_dir = os.path.join('images', self.date_time)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Output folder for saved models
        if self.save_models:
            self.model_out_dir = os.path.join('saved_models', self.date_time)
            if not os.path.exists(self.model_out_dir):
                os.makedirs(self.model_out_dir)

        self.write_metadata_to_JSON()

        # Don't pre-allocate GPU memory; allocate as-needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        # ======= Initialize training ==========
        sys.stdout.flush()
        # plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        self.train(epochs=self.epochs, batch_size=self.batch_size)
        # self.load_model_and_generate_synthetic_images()

# ===============================================================================
# Architecture functions

    # Discriminator layers
    def ck(self, x, k, use_normalization, use_bias):
        x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same', use_bias=use_bias)(x)
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    # First generator layer
    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # Downsampling
    def dk(self, x, k):  # Should have reflection padding
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # Residual block
    def Rk(self, x0):
        k = int(x0.shape[-1])

        # First layer
        x = ReflectionPadding2D((1,1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)

        if self.use_dropout:
            x = Dropout(0.5)(x)

        # Second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # Merge
        x = add([x, x0])

        return x

    # Upsampling
    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias)(x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

#===============================================================================
# Models

    def build_discriminator(self, name=None):
        # Input
        input_img = Input(shape=self.img_shape)

        # Layers 1-4
        x = self.ck(input_img, 64, False, True) #  Instance normalization is not used for this layer)
        x = self.ck(x, 128, True, self.use_bias)
        x = self.ck(x, 256, True, self.use_bias)
        x = self.ck(x, 512, True, self.use_bias)

        # Layer 5: Output
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', use_bias=True)(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)

        if self.discriminator_sigmoid:
            x = Activation('sigmoid')(x)

        return Model(inputs=input_img, outputs=x, name=name)

    def build_generator(self, name=None):
        # Layer 1: Input
        input_img = Input(shape=self.img_shape)
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)

        # Layer 2-3: Downsampling
        x = self.dk(x, 64)
        x = self.dk(x, 128)

        # Layers 4-12: Residual blocks
        for _ in range(4, 13):
            x = self.Rk(x)

        # Layer 13:14: Upsampling
        x = self.uk(x, 64)
        x = self.uk(x, 32)

        # Layer 15: Output
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
        x = Activation('tanh')(x)

        return Model(inputs=input_img, outputs=x, name=name)

#===============================================================================
# Training
    def train(self, epochs, batch_size=1):

        def run_training_batch():

            # ======= Discriminator training ======
            # Generate batch of synthetic images
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)

            # Train discriminators on batch
            D_loss = []
            for _ in range(self.discriminator_iterations):
                D_A_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                D_B_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                D_A_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                D_B_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
                D_A_loss = D_A_loss_real + D_A_loss_synthetic
                D_B_loss = D_B_loss_real + D_B_loss_synthetic
                D_loss.append(D_A_loss + D_B_loss)

            # ======= Generator training ==========
            target_data = [real_images_A, real_images_B, ones, ones]  # Reconstructed images need to match originals, discriminators need to predict ones

            # Train generators on batch
            G_loss = []
            for _ in range(self.generator_iterations):
                G_loss.append(self.G_model.train_on_batch(
                    x=[real_images_A, real_images_B], y=target_data))

            # =====================================

            # Update learning rates
            if self.use_linear_decay and epoch >= self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store training losses
            D_A_losses.append(D_A_loss)
            D_B_losses.append(D_B_loss)
            D_losses.append(D_loss[-1])

            ABA_reconstruction_loss = G_loss[-1][1]
            BAB_reconstruction_loss = G_loss[-1][2]
            reconstruction_loss = ABA_reconstruction_loss + BAB_reconstruction_loss
            G_AB_adversarial_loss = G_loss[-1][3]
            G_BA_adversarial_loss = G_loss[-1][4]

            ABA_reconstruction_losses.append(ABA_reconstruction_loss)
            BAB_reconstruction_losses.append(BAB_reconstruction_loss)
            reconstruction_losses.append(reconstruction_loss)
            G_AB_adversarial_losses.append(G_AB_adversarial_loss)
            G_BA_adversarial_losses.append(G_BA_adversarial_loss)
            G_losses.append(G_loss[-1][0])

            # Print training status
            print('\n')
            print('Epoch ---------------------', epoch, '/', epochs)
            print('Loop index ----------------', loop_index + 1, '/', nr_im_per_epoch)
            if self.discriminator_iterations > 1:
                print('  Discriminator losses:')
                for i in range(self.discriminator_iterations):
                    print('D_loss', D_loss[i])
            if self.generator_iterations > 1:
                print('  Generator losses:')
                for i in range(self.generator_iterations):
                    print('G_loss', G_loss[i])
            print('  Summary:')
            # print('DA_loss:', D_A_loss)
            # print('DB_loss:', D_B_loss)
            print('D_lr:', K.get_value(self.D_A.optimizer.lr))
            print('G_lr', K.get_value(self.G_model.optimizer.lr))
            print('D_loss: ', D_loss[-1])
            print('G_loss: ', G_loss[-1][0])
            print('reconstruction_loss: ', reconstruction_loss)
            self.print_ETA(start_time, epoch, nr_im_per_epoch, loop_index)
            sys.stdout.flush()

            if loop_index % self.tmp_img_update_frequency*self.batch_size == 0:
                # Save temporary images continously
                self.save_tmp_images(real_images_A[0], real_images_B[0],
                                     synthetic_images_A[0], synthetic_images_B[0])

        # ======================================================================
        # Begin training
        # ======================================================================
        if self.save_training_img:
            os.makedirs(os.path.join(self.out_dir, 'train_A'))
            os.makedirs(os.path.join(self.out_dir, 'train_B'))
            os.makedirs(os.path.join(self.out_dir, 'test_A'))
            os.makedirs(os.path.join(self.out_dir, 'test_B'))

        D_A_losses = []
        D_B_losses = []
        D_losses = []

        ABA_reconstruction_losses = []
        BAB_reconstruction_losses = []
        reconstruction_losses = []
        G_AB_adversarial_losses = []
        G_BA_adversarial_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)

        # Labels used for discriminator training
        label_shape = (batch_size,) + self.D_A.output_shape[1:]
        ones = np.ones(shape=label_shape) * self.REAL_LABEL
        zeros = ones * 0

        # Linear learning rate decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        nr_train_im_A = self.A_train.shape[0]
        nr_train_im_B = self.B_train.shape[0]
        nr_im_per_epoch = int(np.ceil(np.max((nr_train_im_A, nr_train_im_B)) / batch_size) * batch_size)

        # Start stopwatch for ETAs
        start_time = time.time()
        timer_started = False

        for epoch in range(1, epochs + 1):
            # random_order_A = np.random.randint(nr_train_im_A, size=nr_im_per_epoch)
            # random_order_B = np.random.randint(nr_train_im_B, size=nr_im_per_epoch)

            random_order_A = np.concatenate((np.random.permutation(nr_train_im_A),
                                             np.random.randint(nr_train_im_A, size=nr_im_per_epoch - nr_train_im_A)))
            random_order_B = np.concatenate((np.random.permutation(nr_train_im_B),
                                             np.random.randint(nr_train_im_B, size=nr_im_per_epoch - nr_train_im_B)))

            # Train on image batch
            for loop_index in range(0, nr_im_per_epoch, batch_size):
                indices_A = random_order_A[loop_index:loop_index + batch_size]
                indices_B = random_order_B[loop_index:loop_index + batch_size]

                real_images_A = self.A_train[indices_A]
                real_images_B = self.B_train[indices_B]

                # Train on image batch
                run_training_batch()

                # Start timer after first (slow) iteration has finished
                if not timer_started:
                    start_time = time.time()
                    timer_started = True

            # Save training images
            if self.save_training_img and epoch % self.save_training_img_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                self.save_epoch_images(epoch)

            # Save model
            if self.save_models and epoch % 20 == 0:
                self.save_model(self.D_A, epoch)
                self.save_model(self.D_B, epoch)
                self.save_model(self.G_A2B, epoch)
                self.save_model(self.G_B2A, epoch)

            # Save training history
            training_history = {
                'DA_losses': D_A_losses,
                'DB_losses': D_B_losses,
                'G_AB_adversarial_losses': G_AB_adversarial_losses,
                'G_BA_adversarial_losses': G_BA_adversarial_losses,
                'ABA_reconstruction_losses': ABA_reconstruction_losses,
                'BAB_reconstruction_losses': BAB_reconstruction_losses,
                'reconstruction_losses': reconstruction_losses,
                'D_losses': D_losses,
                'G_losses': G_losses}
            self.write_loss_data_to_file(training_history)

#===============================================================================
# Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def join_and_save(self, images, save_path):
        # Join images
        image = np.hstack(images)

        # Save images
        if self.channels == 1:
            image = image[:, :, 0]

        # toimage(image, cmin=-1, cmax=1).save(save_path)
        mpimage.imsave(save_path, image, vmin=-1, vmax=1, cmap='gray')

    def save_epoch_images(self, epoch, num_saved_images=1):
        # Save training images
        nr_train_im_A = self.A_train.shape[0]
        nr_train_im_B = self.B_train.shape[0]

        rand_ind_A = np.random.randint(nr_train_im_A)
        rand_ind_B = np.random.randint(nr_train_im_B)

        real_image_A = self.A_train[rand_ind_A]
        real_image_B = self.B_train[rand_ind_B]
        synthetic_image_B = self.G_A2B.predict(real_image_A[np.newaxis])[0]
        synthetic_image_A = self.G_B2A.predict(real_image_B[np.newaxis])[0]
        reconstructed_image_A = self.G_B2A.predict(synthetic_image_B[np.newaxis])[0]
        reconstructed_image_B = self.G_A2B.predict(synthetic_image_A[np.newaxis])[0]

        save_path_A = '{}/train_A/epoch{}.png'.format(self.out_dir, epoch)
        save_path_B = '{}/train_B/epoch{}.png'.format(self.out_dir, epoch)
        if self.paired_data:
            real_image_Ab = self.B_train[rand_ind_A]
            real_image_Ba = self.A_train[rand_ind_B]
            self.join_and_save((real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A), save_path_A)
            self.join_and_save((real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B), save_path_B)
        else:
            self.join_and_save((real_image_A, synthetic_image_B, reconstructed_image_A), save_path_A)
            self.join_and_save((real_image_B, synthetic_image_A, reconstructed_image_B), save_path_B)

        # Save test images
        real_image_A = self.A_test[0]
        real_image_B = self.B_test[0]
        synthetic_image_B = self.G_A2B.predict(real_image_A[np.newaxis])[0]
        synthetic_image_A = self.G_B2A.predict(real_image_B[np.newaxis])[0]
        reconstructed_image_A = self.G_B2A.predict(synthetic_image_B[np.newaxis])[0]
        reconstructed_image_B = self.G_A2B.predict(synthetic_image_A[np.newaxis])[0]

        save_path_A = '{}/test_A/epoch{}.png'.format(self.out_dir, epoch)
        save_path_B = '{}/test_B/epoch{}.png'.format(self.out_dir, epoch)
        if self.paired_data:
            real_image_Ab = self.B_test[0]
            real_image_Ba = self.A_test[0]
            self.join_and_save((real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A), save_path_A)
            self.join_and_save((real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B), save_path_B)
        else:
            self.join_and_save((real_image_A, synthetic_image_B, reconstructed_image_A), save_path_A)
            self.join_and_save((real_image_B, synthetic_image_A, reconstructed_image_B), save_path_B)

    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B[np.newaxis])[0]
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A[np.newaxis])[0]

            real_images = np.vstack((real_image_A, real_image_B))
            synthetic_images = np.vstack((synthetic_image_B, synthetic_image_A))
            reconstructed_images = np.vstack((reconstructed_image_A, reconstructed_image_B))

            save_path = '{}/tmp.png'.format(self.out_dir)
            self.join_and_save((real_images, synthetic_images, reconstructed_images), save_path)
        except: # Ignore if file is open
            pass

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        # max_nr_images = max(len(self.A_train), len(self.B_train))

        nr_train_im_A = self.A_train.shape[0]
        nr_train_im_B = self.B_train.shape[0]
        nr_batches_per_epoch = int(np.ceil(np.max((nr_train_im_A, nr_train_im_B)) / self.batch_size))

        updates_per_epoch_D = 2 * nr_batches_per_epoch
        updates_per_epoch_G = nr_batches_per_epoch
        nr_decay_updates_D = (self.epochs - self.decay_epoch + 1) * updates_per_epoch_D
        nr_decay_updates_G = (self.epochs - self.decay_epoch + 1) * updates_per_epoch_G
        decay_D = self.learning_rate_D / nr_decay_updates_D
        decay_G = self.learning_rate_G / nr_decay_updates_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, nr_im_per_epoch, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * nr_im_per_epoch + loop_index) / self.batch_size
        iterations_total = self.epochs * nr_im_per_epoch / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Elapsed time', passed_time_string, ': ETA in', eta_string)


#===============================================================================
# Save and load

    def save_model(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join('saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_path = '{}/{}_weights_epoch_{}.hdf5'.format(directory, model.name, epoch)
        model.save_weights(weights_path)
        model_path = '{}/{}_model_epoch_{}.json'.format(directory, model.name, epoch)
        model.save_weights(model_path)
        json_string = model.to_json()
        with open(model_path, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def write_loss_data_to_file(self, history):
        keys = sorted(history.keys())
        with open('images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def write_metadata_to_JSON(self):
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save training img interval': self.save_training_img_interval,
            'normalization function': str(self.normalization),
            'lambda_ABA': self.lambda_ABA,
            'lambda_BAB': self.lambda_BAB,
            'lambda_adversarial': self.lambda_adversarial,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of A train examples': len(self.A_train),
            'number of B train examples': len(self.B_train),
            'number of A test examples': len(self.A_test),
            'number of B test examples': len(self.B_test),
            'discriminator sigmoid': self.discriminator_sigmoid,
            'resize convolution': self.use_resize_convolution,
        })

        with open('{}/meta_data.json'.format(self.out_dir), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model):
        path_to_model = os.path.join('generate_images', 'models', '{}.json'.format(model.name))
        path_to_weights = os.path.join('generate_images', 'models', '{}.hdf5'.format(model.name))
        #model = model_from_json(path_to_model)
        model.load_weights(path_to_weights)

    def load_model_and_generate_synthetic_images(self):
        self.load_model_and_weights(self.G_A2B)
        self.load_model_and_weights(self.G_B2A)
        synthetic_images_B = self.G_A2B.predict(self.A_test)
        synthetic_images_A = self.G_B2A.predict(self.B_test)

        def save_image(image, name, domain):
            if self.channels == 1:
                image = image[:, :, 0]
            # image = image.clip(min=0)
            toimage(image, cmin=-1, cmax=1).save(os.path.join(
                'generate_images', 'synthetic_images', domain, name))

        # Test A images
        for i in range(len(synthetic_images_A)):
            # Get the name from the image it was conditioned on
            name = self.testB_image_names[i].strip('.png') + '_synthetic.png'
            synt_A = synthetic_images_A[i]
            save_image(synt_A, name, 'A')

        # Test B images
        for i in range(len(synthetic_images_B)):
            # Get the name from the image it was conditioned on
            name = self.testA_image_names[i].strip('.png') + '_synthetic.png'
            synt_B = synthetic_images_B[i]
            save_image(synt_B, name, 'B')

        print('{} synthetic images have been generated and placed in ./generate_images/synthetic_images'
              .format(len(self.A_test) + len(self.B_test)))


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images


if __name__ == '__main__':
    GAN = CycleGAN()