from keras.layers import Layer, Input, Dropout, Conv3D, Activation, add, BatchNormalization, UpSampling3D, \
    Conv3DTranspose, Flatten
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network

import matplotlib.image as mpimage
import nibabel as nib
import numpy as np
import datetime
import time
import json
import csv
import sys
import os

import keras.backend as K
# dtype='float16'
# K.set_floatx(dtype)
# K.set_epsilon(1e-5)

import tensorflow as tf

from loadData_3D import load_data_3D

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set seed for random numbers
# seedValue = 2323
# os.environ['PYTHONHASHSEED']=str(seedValue)
# import random
# random.seed(seedValue)
# np.random.seed(seedValue)
# tf.set_random_seed(seedValue)

class CycleGAN():
    def __init__(self, args):

        # Parse input arguments
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)  # Select GPU device
        self.volume_folder = os.path.split(args.dataset.rstrip('/'))[-1]
        batch_size = args.batch

        # ======= Data ==========
        print('--- Caching data ---')

        data = load_data_3D(subfolder=self.volume_folder)

        self.channels_A = data["nr_of_channels_A"]
        self.vol_shape_A = data["volume_size_A"] + (self.channels_A,)

        self.channels_B = data["nr_of_channels_B"]
        self.vol_shape_B = data["volume_size_B"] + (self.channels_B,)

        print('volume A shape: ', self.vol_shape_A)
        print('volume B shape: ', self.vol_shape_B)

        self.A_train = data["trainA_volumes"]
        self.B_train = data["trainB_volumes"]
        self.A_test = data["testA_volumes"]
        self.B_test = data["testB_volumes"]

        self.paired_data = True

        # ===== Model parameters ======
        # Training parameters
        self.lambda_ABA = 10.0  # Cyclic loss weight A_2_B
        self.lambda_BAB = 10.0  # Cyclic loss weight B_2_A
        self.lambda_adversarial = 1.0  # Weight for loss from discriminator guess on synthetic volumes
        self.learning_rate_D = 2e-4
        self.learning_rate_G = 2e-4
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.synthetic_pool_size = 50  # Size of volume pools used for training the discriminators
        self.beta_1 = 0.5  # Adam parameter
        self.beta_2 = 0.999  # Adam parameter
        self.batch_size = batch_size  # Number of volumes per batch
        self.epochs = 200  # choose multiples of 20 since the models are saved each 20th epoch

        self.save_models = True  # Save or not the generator and discriminator models
        self.save_training_vol = True  # Save or not example training results or only tmp.png
        self.save_training_vol_interval = 1  # Number of epoch between saves of intermediate training results
        self.tmp_vol_update_frequency = 3  # Number of batches between updates of tmp.png
        self.tmp_img_z_A = self.vol_shape_A[2] // 2
        self.tmp_img_z_B = self.vol_shape_B[2] // 2

        # Architecture parameters
        self.use_instance_normalization = True  # Use instance normalization or batch normalization
        self.use_dropout = False  # Dropout in residual blocks
        self.use_bias = True  # Use bias
        self.use_linear_decay = True  # Linear decay of learning rate, for both discriminators and generators
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start
        self.use_patchgan = True  # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_resize_convolution = False  # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.discriminator_sigmoid = True
        self.generator_residual_blocks = args.resBlocks
        self.base_discirminator_filters = args.baseDiscFilts
        self.base_generator_filters = args.baseGenFilts

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
        D_A = self.build_discriminator(self.vol_shape_A)
        D_B = self.build_discriminator(self.vol_shape_B)

        # Define discriminator models
        volume_A = Input(shape=self.vol_shape_A)
        volume_B = Input(shape=self.vol_shape_B)
        guess_A = D_A(volume_A)
        guess_B = D_B(volume_B)
        self.D_A = Model(inputs=volume_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=volume_B, outputs=guess_B, name='D_B_model')

        # Compile discriminator models
        loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic volumes
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)

        # Use containers to make a static copy of discriminators, used when training the generators
        self.D_A_static = Network(inputs=volume_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=volume_B, outputs=guess_B, name='D_B_static_model')

        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Build generators
        self.G_A2B = self.build_generator(self.vol_shape_A, self.vol_shape_B, name='G_A2B_model')
        self.G_B2A = self.build_generator(self.vol_shape_B, self.vol_shape_A, name='G_B2A_model')

        # Define full CycleGAN model, used for training the generators
        real_A = Input(shape=self.vol_shape_A, name='real_A')
        real_B = Input(shape=self.vol_shape_B, name='real_B')
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
        # if args.tag == None:
        #     # Calculate receptive field
        #     nDiscFiltsStride2 = np.log2(self.D_A.input_shape[1] / self.D_A.output_shape[1])
        #     receptField = int((16 - 3*nDiscFiltsStride2) * 2**nDiscFiltsStride2 + 2**(nDiscFiltsStride2 + 1) - 2)
        #     
        #     # Generate tag
        #     self.tag = '_LR_{}_RL_{}_DF_{}_GF_{}_RF_{}'.format(self.learning_rate_D, self.generator_residual_blocks, self.base_discirminator_filters, self.base_generator_filters, receptField)
        # else:
        #     self.tag = args.tag
        
        if args.tag:
            # Calculate receptive field
            nDiscFiltsStride2 = np.log2(self.D_A.input_shape[1] / self.D_A.output_shape[1])
            receptField = int((16 - 3*nDiscFiltsStride2) * 2**nDiscFiltsStride2 + 2**(nDiscFiltsStride2 + 1) - 2)
            
            # Generate tag
            self.tag = '_LR_{}_RL_{}_DF_{}_GF_{}_RF_{}'.format(self.learning_rate_D, self.generator_residual_blocks, self.base_discirminator_filters, self.base_generator_filters, receptField)
        else:
            self.tag = ''   
            
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + self.volume_folder + self.tag

        # Output folder for run data and volumes
        self.out_dir = os.path.join('runs', self.date_time)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        if self.save_training_vol:
            self.out_dir_volumes = os.path.join(self.out_dir, 'training_volumes')
            if not os.path.exists(self.out_dir_volumes):
                os.makedirs(self.out_dir_volumes)

        # Output folder for saved models
        if self.save_models:
            self.out_dir_models = os.path.join(self.out_dir, 'models')
            if not os.path.exists(self.out_dir_models):
                os.makedirs(self.out_dir_models)

        self.write_metadata_to_JSON()

        # Don't pre-allocate GPU memory; allocate as-needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        # ======= Initialize training ==========
        sys.stdout.flush()
        # plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        self.train(epochs=self.epochs, batch_size=self.batch_size)

# ===============================================================================
# Architecture functions

    # Discriminator layers
    def ck(self, x, k, use_normalization, use_bias, stride):
        x = Conv3D(filters=k, kernel_size=4, strides=stride, padding='same', use_bias=use_bias)(x)
        if use_normalization:
            x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    # First generator layer
    def c7Ak(self, x, k):
        x = Conv3D(filters=k, kernel_size=7, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # Downsampling
    def dk(self, x, k):  # Should have reflection padding
        x = Conv3D(filters=k, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias)(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # Residual block
    def Rk(self, x0):
        k = int(x0.shape[-1])

        # First layer
        x = ReflectionPadding3D((1,1,1))(x0)
        x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)

        if self.use_dropout:
            x = Dropout(0.5)(x)

        # Second layer
        x = ReflectionPadding3D((1, 1, 1))(x)
        x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        # Merge
        x = add([x, x0])

        return x

    # Upsampling
    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling3D(size=(2, 2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding3D((1, 1, 1))(x)
            x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        else:
            x = Conv3DTranspose(filters=k, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias)(x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

#===============================================================================
# Models

    def build_discriminator(self, vol_shape, name=None):
        # Input
        input_vol = Input(shape=vol_shape)

        # Layers 1-4
        x = self.ck(input_vol, self.base_discirminator_filters, False, True, 2) #  Instance normalization is not used for this layer)
        x = self.ck(x, 2*self.base_discirminator_filters, True, self.use_bias, 2)
        x = self.ck(x, 4*self.base_discirminator_filters, True, self.use_bias, 1)
        x = self.ck(x, 8*self.base_discirminator_filters, True, self.use_bias, 1)

        # Layer 5: Output
        if self.use_patchgan:
            x = Conv3D(filters=1, kernel_size=4, strides=1, padding='same', use_bias=True)(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)

        if self.discriminator_sigmoid:
            x = Activation('sigmoid')(x)

        return Model(inputs=input_vol, outputs=x, name=name)

    def build_generator(self, vol_shape_in, vol_shape_out, name=None):
        # Layer 1: Input
        input_vol = Input(shape=vol_shape_in)
        x = ReflectionPadding3D((3, 3, 3))(input_vol)
        x = self.c7Ak(x, self.base_generator_filters)

        # Layer 2-3: Downsampling
        x = self.dk(x, 2*self.base_generator_filters)
        x = self.dk(x, 4*self.base_generator_filters)

        # Layers 4-12: Residual blocks
        for _ in range(4, 4 + self.generator_residual_blocks):
            x = self.Rk(x)

        # Layer 13:14: Upsampling
        x = self.uk(x, 2*self.base_generator_filters)
        x = self.uk(x, self.base_generator_filters)

        # Layer 15: Output
        x = ReflectionPadding3D((3, 3, 3))(x)
        x = Conv3D(filters=vol_shape_out[-1], kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
        x = Activation('tanh')(x)

        return Model(inputs=input_vol, outputs=x, name=name)

#===============================================================================
# Training
    def train(self, epochs, batch_size=1):

        def run_training_batch():

            # ======= Discriminator training ======
            # Generate batch of synthetic volumes
            synthetic_volumes_B = self.G_A2B.predict(real_volumes_A)
            synthetic_volumes_A = self.G_B2A.predict(real_volumes_B)
            synthetic_volumes_B = synthetic_pool_B.query(synthetic_volumes_B)
            synthetic_volumes_A = synthetic_pool_A.query(synthetic_volumes_A)

            # Train discriminators on batch
            D_loss = []
            for _ in range(self.discriminator_iterations):
                D_A_loss_real = self.D_A.train_on_batch(x=real_volumes_A, y=ones)
                D_B_loss_real = self.D_B.train_on_batch(x=real_volumes_B, y=ones)
                D_A_loss_synthetic = self.D_A.train_on_batch(x=synthetic_volumes_A, y=zeros)
                D_B_loss_synthetic = self.D_B.train_on_batch(x=synthetic_volumes_B, y=zeros)
                D_A_loss = D_A_loss_real + D_A_loss_synthetic
                D_B_loss = D_B_loss_real + D_B_loss_synthetic
                D_loss.append(D_A_loss + D_B_loss)

            # ======= Generator training ==========
            target_data = [real_volumes_A, real_volumes_B, ones, ones]  # Reconstructed volumes need to match originals, discriminators need to predict ones

            # Train generators on batch
            G_loss = []
            for _ in range(self.generator_iterations):
                G_loss.append(self.G_model.train_on_batch(
                    x=[real_volumes_A, real_volumes_B], y=target_data))

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
            print('Loop index ----------------', loop_index + 1, '/', nr_vol_per_epoch)
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
            self.print_ETA(start_time, epoch, nr_vol_per_epoch, loop_index)
            sys.stdout.flush()

            if loop_index % self.tmp_vol_update_frequency*self.batch_size == 0:
                # Save temporary images continously
                self.save_tmp_images(real_volumes_A[0], real_volumes_B[0],
                                    synthetic_volumes_A[0], synthetic_volumes_B[0])
                
                

        # ======================================================================
        # Begin training
        # ======================================================================
        if self.save_training_vol:
            os.makedirs(os.path.join(self.out_dir_volumes, 'train_A'))
            os.makedirs(os.path.join(self.out_dir_volumes, 'train_B'))
            os.makedirs(os.path.join(self.out_dir_volumes, 'test_A'))
            os.makedirs(os.path.join(self.out_dir_volumes, 'test_B'))

        D_A_losses = []
        D_B_losses = []
        D_losses = []

        ABA_reconstruction_losses = []
        BAB_reconstruction_losses = []
        reconstruction_losses = []
        G_AB_adversarial_losses = []
        G_BA_adversarial_losses = []
        G_losses = []

        # volume pools used to update the discriminators
        synthetic_pool_A = volumePool(self.synthetic_pool_size)
        synthetic_pool_B = volumePool(self.synthetic_pool_size)

        # Labels used for discriminator training
        label_shape = (batch_size,) + self.D_A.output_shape[1:]
        ones = np.ones(shape=label_shape) * self.REAL_LABEL
        zeros = ones * 0

        # Linear learning rate decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        nr_train_vol_A = self.A_train.shape[0]
        nr_train_vol_B = self.B_train.shape[0]
        nr_vol_per_epoch = int(np.ceil(np.max((nr_train_vol_A, nr_train_vol_B)) / batch_size) * batch_size)

        # Start stopwatch for ETAs
        start_time = time.time()
        timer_started = False

        for epoch in range(1, epochs + 1):
            # random_order_A = np.random.randint(nr_train_vol_A, size=nr_vol_per_epoch)
            # random_order_B = np.random.randint(nr_train_vol_B, size=nr_vol_per_epoch)

            random_order_A = np.concatenate((np.random.permutation(nr_train_vol_A),
                                             np.random.randint(nr_train_vol_A, size=nr_vol_per_epoch - nr_train_vol_A)))
            random_order_B = np.concatenate((np.random.permutation(nr_train_vol_B),
                                             np.random.randint(nr_train_vol_B, size=nr_vol_per_epoch - nr_train_vol_B)))

            # Train on volume batch
            for loop_index in range(0, nr_vol_per_epoch, batch_size):
                indices_A = random_order_A[loop_index:loop_index + batch_size]
                indices_B = random_order_B[loop_index:loop_index + batch_size]

                real_volumes_A = self.A_train[indices_A]
                real_volumes_B = self.B_train[indices_B]

                # Train on volume batch
                run_training_batch()

                # Start timer after first (slow) iteration has finished
                if not timer_started:
                    start_time = time.time()
                    timer_started = True

            # Save training volumes
            if self.save_training_vol and epoch % self.save_training_vol_interval == 0:
                print('\n', '\n', '-------------------------Saving volumes for epoch', epoch, '-------------------------', '\n', '\n')
                self.save_epoch_volumes(epoch)

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
        if image.shape[2] == 1:
            image = image[:, :, 0]
            mpimage.imsave(save_path, image, vmin=-1, vmax=1, cmap='gray')
        else:
            image = (image+1) / 2
            mpimage.imsave(save_path, image)

    def save_epoch_volumes(self, epoch, num_saved_volumes=1):
        # Save training volumes
        nr_train_vol_A = self.A_train.shape[0]
        nr_train_vol_B = self.B_train.shape[0]

        rand_ind_A = np.random.randint(nr_train_vol_A)
        rand_ind_B = np.random.randint(nr_train_vol_B)

        real_volume_A = self.A_train[rand_ind_A]
        real_volume_B = self.B_train[rand_ind_B]
        synthetic_volume_B = self.G_A2B.predict(real_volume_A[np.newaxis])[0]
        synthetic_volume_A = self.G_B2A.predict(real_volume_B[np.newaxis])[0]
        reconstructed_volume_A = self.G_B2A.predict(synthetic_volume_B[np.newaxis])[0]
        reconstructed_volume_B = self.G_A2B.predict(synthetic_volume_A[np.newaxis])[0]

        # # Add dimensions if A and B have different number of channels
        # if self.channels_A == 1 and self.channels_B == 3:
        #     real_volume_A = np.tile(real_volume_A, [1,1,1,3])
        #     synthetic_volume_A = np.tile(synthetic_volume_A, [1,1,1,3])
        #     reconstructed_volume_A = np.tile(reconstructed_volume_A, [1,1,1,3])
        # elif self.channels_B == 1 and self.channels_A == 3:
        #     real_volume_B = np.tile(real_volume_B, [1,1,1,3])
        #     synthetic_volume_B = np.tile(synthetic_volume_B, [1,1,1,3])
        #     reconstructed_volume_B = np.tile(reconstructed_volume_B, [1,1,1,3])

        save_path_A = '{}/train_A/epoch{}.nii.gz'.format(self.out_dir_volumes, epoch)
        save_path_B = '{}/train_B/epoch{}.nii.gz'.format(self.out_dir_volumes, epoch)

        if self.paired_data:
            real_volume_Ab = self.B_train[rand_ind_A]
            real_volume_Ba = self.A_train[rand_ind_B]

            # # Add dimensions if A and B have different number of channels
            # if self.channels_A == 1 and self.channels_B == 3:
            #    real_volume_Ba = np.tile(real_volume_Ba, [1,1,1,3])
            # elif self.channels_B == 1 and self.channels_A == 3:
            #    real_volume_Ab = np.tile(real_volume_Ab, [1,1,1,3])

            image_A = np.hstack((real_volume_Ab, real_volume_A, synthetic_volume_B, reconstructed_volume_A))

            img = nib.Nifti1Image(image_A.astype("float32"), np.eye(4))
            nib.save(img, save_path_A)

            image_B = np.hstack((real_volume_Ba, real_volume_B, synthetic_volume_A, reconstructed_volume_B))

            img = nib.Nifti1Image(image_B.astype("float32"), np.eye(4))
            nib.save(img, save_path_B)
        else:
            image_A = np.hstack((real_volume_A, synthetic_volume_B, reconstructed_volume_A))

            img = nib.Nifti1Image(image_A.astype("float32"), np.eye(4))
            nib.save(img, save_path_A)

            image_B = np.hstack((real_volume_B, synthetic_volume_A, reconstructed_volume_B))

            img = nib.Nifti1Image(image_B.astype("float32"), np.eye(4))
            nib.save(img, save_path_B)

        # Save test volumes
        nr_test_vol_A = self.A_test.shape[0]
        nr_test_vol_B = self.B_test.shape[0]

        rand_ind_A = np.random.randint(nr_test_vol_A)
        rand_ind_B = np.random.randint(nr_test_vol_B)

        real_volume_A = self.A_test[rand_ind_A]
        real_volume_B = self.B_test[rand_ind_B]
        synthetic_volume_B = self.G_A2B.predict(real_volume_A[np.newaxis])[0]
        synthetic_volume_A = self.G_B2A.predict(real_volume_B[np.newaxis])[0]
        reconstructed_volume_A = self.G_B2A.predict(synthetic_volume_B[np.newaxis])[0]
        reconstructed_volume_B = self.G_A2B.predict(synthetic_volume_A[np.newaxis])[0]

        # # Add dimensions if A and B have different number of channels
        # if self.channels_A == 1 and self.channels_B == 3:
        #     real_volume_A = np.tile(real_volume_A, [1,1,3])
        #     synthetic_volume_A = np.tile(synthetic_volume_A, [1,1,3])
        #     reconstructed_volume_A = np.tile(reconstructed_volume_A, [1,1,3])
        # elif self.channels_B == 1 and self.channels_A == 3:
        #     real_volume_B = np.tile(real_volume_B, [1,1,3])
        #     synthetic_volume_B = np.tile(synthetic_volume_B, [1,1,3])
        #     reconstructed_volume_B = np.tile(reconstructed_volume_B, [1,1,3])

        save_path_A = '{}/test_A/epoch{}.nii.gz'.format(self.out_dir_volumes, epoch)
        save_path_B = '{}/test_B/epoch{}.nii.gz'.format(self.out_dir_volumes, epoch)

        if self.paired_data:
            real_volume_Ab = self.B_test[rand_ind_A] 
            real_volume_Ba = self.A_test[rand_ind_B]

            # # Add dimensions if A and B have different number of channels
            # if self.channels_A == 1 and self.channels_B == 3:
            #     real_volume_Ba = np.tile(real_volume_Ba, [1,1,1,3])
            # elif self.channels_B == 1 and self.channels_A == 3:
            #     real_volume_Ab = np.tile(real_volume_Ab, [1,1,1,3])

            image_A = np.hstack((real_volume_Ab, real_volume_A, synthetic_volume_B, reconstructed_volume_A))

            img = nib.Nifti1Image(image_A.astype("float32"), np.eye(4))
            nib.save(img, save_path_A)

            image_B = np.hstack((real_volume_Ba, real_volume_B, synthetic_volume_A, reconstructed_volume_B))

            img = nib.Nifti1Image(image_B.astype("float32"), np.eye(4))
            nib.save(img, save_path_B)
        else:
            image_A = np.hstack((real_volume_A, synthetic_volume_B, reconstructed_volume_A))

            img = nib.Nifti1Image(image_A.astype("float32"), np.eye(4))
            nib.save(img, save_path_A)

            image_B = np.hstack((real_volume_B, synthetic_volume_A, reconstructed_volume_B))

            img = nib.Nifti1Image(image_B.astype("float32"), np.eye(4))
            nib.save(img, save_path_B)

    def save_tmp_images(self, real_volume_A, real_volume_B, synthetic_volume_A, synthetic_volume_B):
        try:
            reconstructed_volume_A = self.G_B2A.predict(synthetic_volume_B[np.newaxis])[0]
            reconstructed_volume_B = self.G_A2B.predict(synthetic_volume_A[np.newaxis])[0]

            real_image_A = real_volume_A[:,:,self.tmp_img_z_A,:]
            real_image_B = real_volume_B[:,:,self.tmp_img_z_B,:]
            synthetic_image_A = synthetic_volume_A[:,:,self.tmp_img_z_A,:]
            synthetic_image_B = synthetic_volume_B[:,:,self.tmp_img_z_B,:]
            reconstructed_image_A = reconstructed_volume_A[:,:,self.tmp_img_z_A,:]
            reconstructed_image_B = reconstructed_volume_B[:,:,self.tmp_img_z_B,:]

            # Add dimensions if A and B have different number of channels
            if self.channels_A == 1 and self.channels_B == 3:
                real_image_A = np.tile(real_image_A, [1,1,3])
                synthetic_image_A = np.tile(synthetic_image_A, [1,1,3])
                reconstructed_image_A = np.tile(reconstructed_image_A, [1,1,3])
            elif self.channels_B == 1 and self.channels_A == 3:
                real_image_B = np.tile(real_image_B, [1,1,3])
                synthetic_image_B = np.tile(synthetic_image_B, [1,1,3])
                reconstructed_image_B = np.tile(reconstructed_image_B, [1,1,3])

            real_images = np.vstack((real_image_A, real_image_B))
            synthetic_images = np.vstack((synthetic_image_B, synthetic_image_A))
            reconstructed_images = np.vstack((reconstructed_image_A, reconstructed_image_B))

            save_path = '{}/tmp.png'.format(self.out_dir)
            self.join_and_save((real_images, synthetic_images, reconstructed_images), save_path)
        except: # Ignore if file is open
            pass

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        # max_nr_volumes = max(len(self.A_train), len(self.B_train))

        nr_train_vol_A = self.A_train.shape[0]
        nr_train_vol_B = self.B_train.shape[0]
        nr_batches_per_epoch = int(np.ceil(np.max((nr_train_vol_A, nr_train_vol_B)) / self.batch_size))

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

    def print_ETA(self, start_time, epoch, nr_vol_per_epoch, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * nr_vol_per_epoch + loop_index) / self.batch_size
        iterations_total = self.epochs * nr_vol_per_epoch / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Elapsed time', passed_time_string, ': ETA in', eta_string)


#===============================================================================
# Save and load

    def save_model(self, model, epoch):
        weights_path = '{}/{}_epoch_{}.hdf5'.format(self.out_dir_models, model.name, epoch)
        model.save_weights(weights_path)
        
        model_path = '{}/{}_epoch_{}.json'.format(self.out_dir_models, model.name, epoch)
        model_json_string = model.to_json()
        with open(model_path, 'w') as outfile:
            outfile.write(model_json_string)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def write_loss_data_to_file(self, history):
        keys = sorted(history.keys())
        with open('runs/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def write_metadata_to_JSON(self):
        # Save metadata
        metadata = {
            'vol shape_A: height,width,depth,channels': self.vol_shape_A,
            'vol shape_B: height,width,depth,channels': self.vol_shape_B,
            'batch size': self.batch_size,
            'save training vol interval': self.save_training_vol_interval,
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
            'tag': self.tag,
            'volume_folder': self.volume_folder 
        }

        with open('{}/metadata.json'.format(self.out_dir), 'w') as outfile:
            json.dump(metadata, outfile, sort_keys=True)

# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3] + 2 * self.padding[2], s[4])

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], 'REFLECT')
        
    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class volumePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_vols = 0
            self.volumes = []

    def query(self, volumes):
        if self.pool_size == 0:
            return volumes
        return_volumes = []
        for volume in volumes:
            if len(volume.shape) == 4:
                volume = volume[np.newaxis, :, :, :, :]

            if self.num_vols < self.pool_size:  # fill up the volume pool
                self.num_vols = self.num_vols + 1
                if len(self.volumes) == 0:
                    self.volumes = volume
                else:
                    self.volumes = np.vstack((self.volumes, volume))

                if len(return_volumes) == 0:
                    return_volumes = volume
                else:
                    return_volumes = np.vstack((return_volumes, volume))

            else:  # 50% chance that we replace an old synthetic volume
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.volumes[random_id, :, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :, :]
                    self.volumes[random_id, :, :, :, :] = volume[0, :, :, :, :]
                    if len(return_volumes) == 0:
                        return_volumes = tmp
                    else:
                        return_volumes = np.vstack((return_volumes, tmp))
                else:
                    if len(return_volumes) == 0:
                        return_volumes = volume
                    else:
                        return_volumes = np.vstack((return_volumes, volume))

        return return_volumes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='name of the dataset on which to run CycleGAN (stored in data/)')
    
    parser.add_argument('-r', '--resBlocks', type=int, default=9, help='number of residual blocks used in the generators (default: 9)')
    parser.add_argument('-df', '--baseDiscFilts', type=int, default=64, help='number of filters in the first discriminator layer (default: 64)')
    parser.add_argument('-gf', '--baseGenFilts', type=int, default=32, help='number of filters in the first generator layer (default: 32)')
    parser.add_argument('-b', '--batch', type=int, default=5, help='batch size to use during training (default: 5)')
    
    parser.add_argument('-g', '--gpu', type=int, default=0, help='ID of GPU on which to run (default: 0)')
    # parser.add_argument('-t', '--tag', help='tag to remember specific settings for each training session (default: generate automatically)')
    parser.add_argument('-t', '--tag', action='store_true', help='tag to remember specific settings for each training session (default: no tag)')
    
    args = parser.parse_args()
    
    CycleGAN(args)

