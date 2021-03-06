import glob
import os
import sys
import numpy as np

import nibabel as nib

from progress.bar import Bar

def load_data_3D(subfolder=''):

    dataset_path = os.path.join('data', subfolder)
    if not os.path.isdir(dataset_path):
        sys.exit(' Dataset ' + subfolder + ' does not exist')

    # volume paths
    trainA_path = os.path.join(dataset_path, 'trainA')
    trainB_path = os.path.join(dataset_path, 'trainB')
    testA_path = os.path.join(dataset_path, 'testA')
    testB_path = os.path.join(dataset_path, 'testB')

    # volume file names
    trainA_volume_names = sorted(glob.glob(os.path.join(trainA_path,'*.nii.gz')))
    trainB_volume_names = sorted(glob.glob(os.path.join(trainB_path,'*.nii.gz')))
    testA_volume_names = sorted(glob.glob(os.path.join(testA_path,'*.nii.gz')))
    testB_volume_names = sorted(glob.glob(os.path.join(testB_path,'*.nii.gz')))

    trainA_volume_names = [os.path.basename(x) for x in trainA_volume_names]
    trainB_volume_names = [os.path.basename(x) for x in trainB_volume_names]
    testA_volume_names = [os.path.basename(x) for x in testA_volume_names]
    testB_volume_names = [os.path.basename(x) for x in testB_volume_names]

    # Examine one volume to get size and number of channels
    vol_test_A = nib.load(os.path.join(trainA_path, trainA_volume_names[0]))
    vol_test_B = nib.load(os.path.join(trainB_path, trainB_volume_names[0]))    

    if len(vol_test_A.shape) == 3:
        volume_size_A = vol_test_A.shape
        nr_of_channels_A = 1
    else:
        volume_size_A = vol_test_A.shape[0:-1]
        nr_of_channels_A = vol_test_A.shape[-1]
        
    if len(vol_test_B.shape) == 3:
        volume_size_B = vol_test_B.shape
        nr_of_channels_B = 1
    else:
        volume_size_B = vol_test_B.shape[0:-1]
        nr_of_channels_B = vol_test_B.shape[-1]

    trainA_volumes = create_volume_array(trainA_volume_names, trainA_path, volume_size_A, nr_of_channels_A)
    trainB_volumes = create_volume_array(trainB_volume_names, trainB_path, volume_size_B, nr_of_channels_B)
    testA_volumes = create_volume_array(testA_volume_names, testA_path, volume_size_A, nr_of_channels_A)
    testB_volumes = create_volume_array(testB_volume_names, testB_path, volume_size_B, nr_of_channels_B)

    # Normalize input volumes
    A995quant = np.quantile(np.vstack((trainA_volumes, testA_volumes)), 0.995)
    B995quant = np.quantile(np.vstack((trainB_volumes, testB_volumes)), 0.995)

    normConstant = np.max((A995quant, B995quant)) / 2

    return {"volume_size_A": volume_size_A, "nr_of_channels_A": nr_of_channels_A,
            "volume_size_B": volume_size_B, "nr_of_channels_B": nr_of_channels_B,
            "trainA_volumes": np.clip(trainA_volumes/normConstant - 1, -1, 1),
            "trainB_volumes": np.clip(trainB_volumes/normConstant - 1, -1, 1),
            "testA_volumes": np.clip(testA_volumes/normConstant - 1, -1, 1),
            "testB_volumes": np.clip(testB_volumes/normConstant - 1, -1, 1)}

def load_test_data_3D(subfolder=''):

    dataset_path = os.path.join('data', subfolder)
    if not os.path.isdir(dataset_path):
        sys.exit(' Dataset ' + subfolder + ' does not exist')

    # volume paths
    trainA_path = os.path.join(dataset_path, 'trainA')
    trainB_path = os.path.join(dataset_path, 'trainB')
    testA_path = os.path.join(dataset_path, 'testA')
    testB_path = os.path.join(dataset_path, 'testB')

    # volume file names
    trainA_volume_names = sorted(glob.glob(os.path.join(trainA_path,'*.nii.gz')))
    trainB_volume_names = sorted(glob.glob(os.path.join(trainB_path,'*.nii.gz')))
    testA_volume_names = sorted(glob.glob(os.path.join(testA_path,'*.nii.gz')))
    testB_volume_names = sorted(glob.glob(os.path.join(testB_path,'*.nii.gz')))

    trainA_volume_names = [os.path.basename(x) for x in trainA_volume_names]
    trainB_volume_names = [os.path.basename(x) for x in trainB_volume_names]
    testA_volume_names = [os.path.basename(x) for x in testA_volume_names]
    testB_volume_names = [os.path.basename(x) for x in testB_volume_names]

    # Examine one volume to get size, number of channels, and header
    vol_test_A = nib.load(os.path.join(trainA_path, trainA_volume_names[0]))
    vol_test_B = nib.load(os.path.join(trainB_path, trainB_volume_names[0]))    

    if len(vol_test_A.shape) == 3:
        volume_size_A = vol_test_A.shape
        nr_of_channels_A = 1
    else:
        volume_size_A = vol_test_A.shape[0:-1]
        nr_of_channels_A = vol_test_A.shape[-1]
        
    if len(vol_test_B.shape) == 3:
        volume_size_B = vol_test_B.shape
        nr_of_channels_B = 1
    else:
        volume_size_B = vol_test_B.shape[0:-1]
        nr_of_channels_B = vol_test_B.shape[-1]

    header_A = vol_test_A.get_header()
    header_B = vol_test_B.get_header()

    trainA_volumes = create_volume_array(trainA_volume_names, trainA_path, volume_size_A, nr_of_channels_A)
    trainB_volumes = create_volume_array(trainB_volume_names, trainB_path, volume_size_B, nr_of_channels_B)
    testA_volumes = create_volume_array(testA_volume_names, testA_path, volume_size_A, nr_of_channels_A)
    testB_volumes = create_volume_array(testB_volume_names, testB_path, volume_size_B, nr_of_channels_B)

    # Normalize input volumes
    A995quant = np.quantile(np.vstack((trainA_volumes, testA_volumes)), 0.995)
    B995quant = np.quantile(np.vstack((trainB_volumes, testB_volumes)), 0.995)

    normConstant = np.max((A995quant, B995quant)) / 2
    
    return {"volume_size_A": volume_size_A, "nr_of_channels_A": nr_of_channels_A,
            "volume_size_B": volume_size_B, "nr_of_channels_B": nr_of_channels_B,
            "header_A": header_A, "header_B": header_B,
            "testA_volumes": np.clip(testA_volumes/normConstant - 1, -1, 1),
            "testB_volumes": np.clip(testB_volumes/normConstant - 1, -1, 1),
            "testA_volume_names": testA_volume_names,
            "testB_volume_names": testB_volume_names,
            "norm_const": normConstant}


def create_volume_array(volume_list, volume_path, volume_size, nr_of_channels):
    bar = Bar('Loading...', max=len(volume_list))

    # Define volume array
    volume_array = np.empty((len(volume_list),) + (volume_size) + (nr_of_channels,), dtype="float32")
    i = 0
    for volume_name in volume_list:
        
        # Load volume and convert into np.array
        volume = nib.load(os.path.join(volume_path, volume_name)).get_fdata()  # Normalized to [0,1]
        volume = volume.astype("float32")            

        # Add third dimension if volume is 2D
        if nr_of_channels == 1:  # Gray scale volume -> MR volume
            volume = volume[:, :, :, np.newaxis]
            
        # Add volume to array
        volume_array[i, :, :, :, :] = volume
        i += 1
        bar.next()
    bar.finish()
    
    return volume_array

if __name__ == '__main__':
    load_data_3D()

