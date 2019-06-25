from keras.layers import Layer, Input, Dropout, Conv2D, Activation, add, BatchNormalization, UpSampling2D, \
    Conv2DTranspose, Flatten
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network

import matplotlib.image as mpimage
import numpy as np
import glob
import datetime
import time
import json
import csv
import sys
import os

import keras.backend as K
import tensorflow as tf

from loadData import load_test_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# np.random.seed(seed=12345)

class CycleGAN():
    def __init__(self, model_subfolder):

        self.model_subfolder = os.path.split(model_subfolder)[-1]
        self.model_path = os.path.join('saved_models', self.model_subfolder)
        if not os.path.isdir(self.model_path):
            sys.exit(' Model ' + self.model_subfolder + ' does not exist')
            
        # Custom layers
        self.custom_objects = {'InstanceNormalization':InstanceNormalization,
                          'ReflectionPadding2D':ReflectionPadding2D}

        image_folder = self.model_subfolder[16:]
        
        # ======= Data ==========
        print('--- Caching data ---')

        data = load_test_data(subfolder=image_folder)

        self.channels_A = data["nr_of_channels_A"]
        self.img_shape_A = data["image_size_A"] + (self.channels_A,)

        self.channels_B = data["nr_of_channels_B"]
        self.img_shape_B = data["image_size_B"] + (self.channels_B,)

        print('Image A shape: ', self.img_shape_A)
        print('Image B shape: ', self.img_shape_B)

        self.A_test = data["testA_images"]
        self.testA_image_names = data["testA_image_names"]

        # Don't pre-allocate GPU memory; allocate as-needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        # ======= Generate images ==========
        sys.stdout.flush()
        self.generate_synthetic_images()

#===============================================================================
# Save and load

    def load_model_and_weights(self, path_to_model, path_to_weights):
        with open(path_to_model, 'r') as infile:
            model_json_string = infile.read()
        
        model = model_from_json(model_json_string, self.custom_objects)
        model.load_weights(path_to_weights)
        return model

    def save_image(self, image, save_path):
        if image.shape[2] == 1:
            image = image[:, :, 0]
            mpimage.imsave(save_path, image, vmin=-1, vmax=1, cmap='gray')
        else:
            image = (image+1) / 2
            mpimage.imsave(save_path, image)

    def generate_synthetic_images(self):        
        # Find all generator models in folder                                  
        generator_models = sorted(glob.glob(os.path.join(self.model_path,'G_A2B*.json')))
        
        for path_to_model in generator_models:
            path_to_weights = path_to_model[:-5] + '.hdf5'
            
            self.G_A2B = self.load_model_and_weights(path_to_model, path_to_weights)
            
            synthetic_images_B = self.G_A2B.predict(self.A_test)

            epoch_string_start = path_to_model.find('epoch')
            epoch_string = path_to_model[epoch_string_start:-5]
            
            out_dir = save_path = os.path.join('generated_images', self.model_subfolder, epoch_string, 'A2B')
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            # Test B images
            for i in range(synthetic_images_B.shape[0]):
                synt_B = synthetic_images_B[i]
                
                # Get the name from the image it was conditioned on
                out_name = self.testA_image_names[i][:-4] + '_synthetic.png'
                save_path = os.path.join(out_dir, out_name)
                
                self.save_image(synt_B, save_path)
            
            print('{} synthetic images generated, {}'.format(synthetic_images_B.shape[0], epoch_string))

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


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit(' Usage: python CycleGAN_generate.py model_subfolder')
    
    CycleGAN(sys.argv[1])