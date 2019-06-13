import os
import numpy as np

# from PIL import Image
import matplotlib.image as mpimage
from progress.bar import Bar

def load_data(subfolder='', generator=False):

    # Image paths
    trainA_path = os.path.join('data', subfolder, 'trainA')
    trainB_path = os.path.join('data', subfolder, 'trainB')
    testA_path = os.path.join('data', subfolder, 'testA')
    testB_path = os.path.join('data', subfolder, 'testB')

    # Image file names
    trainA_image_names = sorted(os.listdir(trainA_path))
    trainB_image_names = sorted(os.listdir(trainB_path))
    testA_image_names = sorted(os.listdir(testA_path))
    testB_image_names = sorted(os.listdir(testB_path))

    # Examine one image to get size and number of channels
    im_test = mpimage.imread(os.path.join(trainA_path, trainA_image_names[0]))
    # im_test = np.array(Image.open(os.path.join(trainA_path, trainA_image_names[0])))
    

    if len(im_test.shape) == 2:
        image_size = im_test.shape
        nr_of_channels = 1
    else:
        image_size = im_test.shape[0:-1]
        nr_of_channels = im_test.shape[-1]

    trainA_images = create_image_array(trainA_image_names, trainA_path, image_size, nr_of_channels)
    trainB_images = create_image_array(trainB_image_names, trainB_path, image_size, nr_of_channels)
    testA_images = create_image_array(testA_image_names, testA_path, image_size, nr_of_channels)
    testB_images = create_image_array(testB_image_names, testB_path, image_size, nr_of_channels)
    
    return {"image_size": image_size, "nr_of_channels": nr_of_channels,
            "trainA_images": trainA_images, "trainB_images": trainB_images,
            "testA_images": testA_images, "testB_images": testB_images,
            "trainA_image_names": trainA_image_names,
            "trainB_image_names": trainB_image_names,
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
