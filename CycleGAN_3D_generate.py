from keras.layers import Layer, Input, Dropout, Conv2D, Activation, add, BatchNormalization, UpSampling2D, \
    Conv2DTranspose, Flatten
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network

import nibabel as nib
from progress.bar import Bar
import numpy as np
import glob
import sys
import os

import keras.backend as K
import tensorflow as tf

from loadData_3D import load_test_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# np.random.seed(seed=12345)

class CycleGAN():
    def __init__(self, args):

        # Parse input arguments
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)  # Select GPU device

        self.model_subfolder = os.path.split(args.model.rstrip('/'))[-1]
        self.model_path = os.path.join('saved_models', self.model_subfolder)
        if not os.path.isdir(self.model_path):
            sys.exit(' Model ' + self.model_subfolder + ' does not exist')

        if args.data:  # If data folder is provided, use it
            data_folder = args.data = os.path.split(args.data.rstrip('/'))[-1]
            self.out_dir = os.path.join('generated_volumes', self.model_subfolder + '_data_' + data_folder)
        else:  # If data folder is not provided, guess from model name
            data_folder = self.model_subfolder[16:]
            self.out_dir = os.path.join('generated_volumes', self.model_subfolder)

        if args.epochs:  # If epochs are provided, make a list
            try:
                self.epochs = [int(x) for x in args.epochs.split(',')]
            except:
                sys.exit(' Epochs must be a comma-separated list of integers')
        else:  # If epochs are not provided, use all available
            self.epochs = None

        if not args.A2B and not args.B2A:  # If no argument is passed, generate A2B and B2A
            self.generate_A2B = True
            self.generate_B2A = True
        else:  # If either argument is passed, generate only A2B or B2A
            self.generate_A2B = args.A2B
            self.generate_B2A = args.B2A
            
        # Custom layers
        self.custom_objects = {'InstanceNormalization':InstanceNormalization,
                          'ReflectionPadding3D':ReflectionPadding3D}

        # ======= Data ==========
        print('--- Caching data ---')

        data = load_test_data(subfolder=data_folder)

        self.channels_A = data["nr_of_channels_A"]
        self.vol_shape_A = data["volume_size_A"] + (self.channels_A,)

        self.channels_B = data["nr_of_channels_B"]
        self.vol_shape_B = data["volume_size_B"] + (self.channels_B,)

        print('Volume A shape: ', self.vol_shape_A)
        print('Volume B shape: ', self.vol_shape_B)

        self.A_test = data["testA_volumes"]
        self.testA_volume_names = data["testA_volume_names"]
        
        self.B_test = data["testB_volumes"]
        self.testB_volume_names = data["testB_volume_names"]

        self.norm_const = data["norm_const"]

        # Don't pre-allocate GPU memory; allocate as-needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        # ======= Generate volumes ==========
        sys.stdout.flush()
        self.generate_synthetic_volumes()

#===============================================================================
# Save and load

    def load_model_and_weights(self, path_to_model, path_to_weights):
        with open(path_to_model, 'r') as infile:
            model_json_string = infile.read()
        
        model = model_from_json(model_json_string, self.custom_objects)
        model.load_weights(path_to_weights)
        return model

    def save_volume(self, volume, save_path):
        volume = (volume + 1) * self.norm_const  # Undo normalization
        img = nib.Nifti1Image(volume.astype("float32"), np.eye(4))
        nib.save(img, save_path)

        # if image.shape[2] == 1:
        #     image = image[:, :, 0]
        #     mpimage.imsave(save_path, image, vmin=-1, vmax=1, cmap='gray')
        # else:
        #     image = (image+1) / 2
        #     mpimage.imsave(save_path, image)

    def generate_synthetic_volumes(self):
        print()
        
        if self.generate_A2B:
            # Find all A2B generator models in folder
            generator_models = sorted(glob.glob(os.path.join(self.model_path,'G_A2B*.json')))
            
            for path_to_model in generator_models:
                epoch_string_start = path_to_model.find('epoch')
                epoch_string = path_to_model[epoch_string_start:-5]
                epoch = int(epoch_string[6:])

                if not self.epochs == None and epoch not in self.epochs:  # Check if current epoch is in list
                    continue
                    
                path_to_weights = path_to_model[:-5] + '.hdf5'
                
                print('G_A2B, {}, generating volumes... '.format(epoch_string))
                self.G_A2B = self.load_model_and_weights(path_to_model, path_to_weights)
                
                synthetic_volumes_B = self.G_A2B.predict(self.A_test)
                    
                out_dir = save_path = os.path.join(self.out_dir, epoch_string, 'A2B')
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
    
                # Test B volumes
                bar = Bar('saving...', max=synthetic_volumes_B.shape[0])
                for i in range(synthetic_volumes_B.shape[0]):
                    synt_B = synthetic_volumes_B[i]
                    
                    # Get the name from the volume it was conditioned on
                    out_name = self.testA_volume_names[i][:-7] + '_synthetic.nii.gz'
                    save_path = os.path.join(out_dir, out_name)
                    
                    self.save_volume(synt_B, save_path)
                    bar.next()
                bar.finish()

        if self.generate_B2A:
            # Find all B2A generator models in folder
            generator_models = sorted(glob.glob(os.path.join(self.model_path,'G_B2A*.json')))
            
            for path_to_model in generator_models:
                epoch_string_start = path_to_model.find('epoch')
                epoch_string = path_to_model[epoch_string_start:-5]
                epoch = int(epoch_string[6:])
                            
                if not self.epochs == None and epoch not in self.epochs:  # Check if current epoch is in list
                    continue
                
                path_to_weights = path_to_model[:-5] + '.hdf5'

                print('G_B2A, {}, generating volumes... '.format(epoch_string))
                self.G_B2A = self.load_model_and_weights(path_to_model, path_to_weights)
                
                synthetic_volumes_A = self.G_B2A.predict(self.B_test)
                
                out_dir = save_path = os.path.join(self.out_dir, epoch_string, 'B2A')
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
    
                # Test A volumes
                bar = Bar('saving...', max=synthetic_volumes_A.shape[0])
                for i in range(synthetic_volumes_A.shape[0]):
                    synt_A = synthetic_volumes_A[i]
                    
                    # Get the name from the volume it was conditioned on
                    out_name = self.testB_volume_names[i][:-7] + '_synthetic.nii.gz'
                    save_path = os.path.join(out_dir, out_name)
                    
                    self.save_volume(synt_A, save_path)
                    bar.next()
                bar.finish()

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('model', help='name of the model on which to run CycleGAN, stored in saved_models/')
    parser.add_argument('-d', '--data', help='dataset on which to apply the model, stored in data/ (default: guessed from the model name)')
    parser.add_argument('-e', '--epochs', help='comma-separated list of model epochs to use (default: all available)')
    
    parser.add_argument('-g', '--gpu', type=int, default=0, help='ID of GPU on which to run (default: 0)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--A2B', action='store_true', help='apply only A2B conversion')
    group.add_argument('-b', '--B2A', action='store_true', help='apply only B2A conversion')

    args = parser.parse_args()
    
    CycleGAN(args)