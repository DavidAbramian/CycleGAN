import glob
import os
import sys
import numpy as np

# from PIL import Image
import matplotlib.image as mpimage
from progress.bar import Bar

def load_data(subfolder=''):

    dataset_path = os.path.join('data', subfolder)
    if not os.path.isdir(dataset_path):
        sys.exit(' Dataset ' + subfolder + ' does not exist')

    # Image paths
    trainA_path = os.path.join(dataset_path, 'trainA')
    trainB_path = os.path.join(dataset_path, 'trainB')
    testA_path = os.path.join(dataset_path, 'testA')
    testB_path = os.path.join(dataset_path, 'testB')

    # Image file names
    trainA_image_names = sorted(glob.glob(os.path.join(trainA_path,'*.png')))
    trainB_image_names = sorted(glob.glob(os.path.join(trainB_path,'*.png')))
    testA_image_names = sorted(glob.glob(os.path.join(testA_path,'*.png')))
    testB_image_names = sorted(glob.glob(os.path.join(testB_path,'*.png')))

    trainA_image_names = [os.path.basename(x) for x in trainA_image_names]
    trainB_image_names = [os.path.basename(x) for x in trainB_image_names]
    testA_image_names = [os.path.basename(x) for x in testA_image_names]
    testB_image_names = [os.path.basename(x) for x in testB_image_names]

    # Examine one image to get size and number of channels
    im_test_A = mpimage.imread(os.path.join(trainA_path, trainA_image_names[0]))
    im_test_B = mpimage.imread(os.path.join(trainB_path, trainB_image_names[0]))    

    if len(im_test_A.shape) == 2:
        image_size_A = im_test_A.shape
        nr_of_channels_A = 1
    else:
        image_size_A = im_test_A.shape[0:-1]
        nr_of_channels_A = im_test_A.shape[-1]
        
    if len(im_test_B.shape) == 2:
        image_size_B = im_test_B.shape
        nr_of_channels_B = 1
    else:
        image_size_B = im_test_B.shape[0:-1]
        nr_of_channels_B = im_test_B.shape[-1]

    trainA_images = create_image_array(trainA_image_names, trainA_path, image_size_A, nr_of_channels_A)
    trainB_images = create_image_array(trainB_image_names, trainB_path, image_size_B, nr_of_channels_B)
    testA_images = create_image_array(testA_image_names, testA_path, image_size_A, nr_of_channels_A)
    testB_images = create_image_array(testB_image_names, testB_path, image_size_B, nr_of_channels_B)
    
    return {"image_size_A": image_size_A, "nr_of_channels_A": nr_of_channels_A,
            "image_size_B": image_size_B, "nr_of_channels_B": nr_of_channels_B,
            "trainA_images": trainA_images, "trainB_images": trainB_images,
            "testA_images": testA_images, "testB_images": testB_images}

def load_test_data(subfolder=''):

    dataset_path = os.path.join('data', subfolder)
    if not os.path.isdir(dataset_path):
        sys.exit(' Dataset ' + subfolder + ' does not exist')

    # Image paths
    testA_path = os.path.join(dataset_path, 'testA')
    testB_path = os.path.join(dataset_path, 'testB')

    # Image file names
    testA_image_names = sorted(glob.glob(os.path.join(testA_path,'*.png')))
    testB_image_names = sorted(glob.glob(os.path.join(testB_path,'*.png')))

    testA_image_names = [os.path.basename(x) for x in testA_image_names]
    testB_image_names = [os.path.basename(x) for x in testB_image_names]

    # Examine one image to get size and number of channels
    im_test_A = mpimage.imread(os.path.join(testA_path, testA_image_names[0]))
    im_test_B = mpimage.imread(os.path.join(testB_path, testB_image_names[0]))    
    

    if len(im_test_A.shape) == 2:
        image_size_A = im_test_A.shape
        nr_of_channels_A = 1
    else:
        image_size_A = im_test_A.shape[0:-1]
        nr_of_channels_A = im_test_A.shape[-1]
        
    if len(im_test_B.shape) == 2:
        image_size_B = im_test_B.shape
        nr_of_channels_B = 1
    else:
        image_size_B = im_test_B.shape[0:-1]
        nr_of_channels_B = im_test_B.shape[-1]

    testA_images = create_image_array(testA_image_names, testA_path, image_size_A, nr_of_channels_A)
    testB_images = create_image_array(testB_image_names, testB_path, image_size_B, nr_of_channels_B)
    
    return {"image_size_A": image_size_A, "nr_of_channels_A": nr_of_channels_A,
            "image_size_B": image_size_B, "nr_of_channels_B": nr_of_channels_B,
            "testA_images": testA_images, "testB_images": testB_images,
            "testA_image_names": testA_image_names,
            "testB_image_names": testB_image_names}

def create_image_array(image_list, image_path, image_size, nr_of_channels):
    bar = Bar('Loading...', max=len(image_list))

    # Define image array
    image_array = np.empty((len(image_list),) + (image_size) + (nr_of_channels,))
    i = 0
    for image_name in image_list:
        # If file is image...
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            # Load image and convert into np.array
            image = mpimage.imread(os.path.join(image_path, image_name))  # Normalized to [0,1]
            # image = np.array(Image.open(os.path.join(image_path, image_name)))
            
            # Add third dimension if image is 2D
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = image[:, :, np.newaxis]
            
            # Normalize image with (max 8 bit value - 1)
            image = image * 2 - 1
            # image = image / 127.5 - 1
            
            # Add image to array
            image_array[i, :, :, :] = image
            i += 1
            bar.next()
    bar.finish()

    return image_array

if __name__ == '__main__':
    load_data()
