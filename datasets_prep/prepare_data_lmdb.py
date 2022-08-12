"""
Convert png in dataset folder into lmdb object to be readable for certain networks
"""
# lmdbconverter.py - https://www.pankesh.com/_posts/2019-05-18-efficiently-storing-and-retrieving-image-datasets.html

import os
import glob
import lmdb
import numpy as np
from itertools import tee
from typing import Generator, Any
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import idx2numpy

class LMDB_Image:
    def __init__(self, image, label=None):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        if label is not None:
            self.label = label
        self.image = image.tobytes()

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels), self.label

def store_many_lmdb(images, labels, lmdb_dir="/data/lrudden/diffusion_TM/dataset/train_lmdb"):
    """ Stores an array of images to LMDB.
        https://realpython.com/storing-images-in-python/#storing-to-lmdb
        Parameters:
        ---------------
        images       images array, (N, 64, 64, 1) to be stored
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    #env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)
    env = lmdb.open(str(lmdb_dir), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = LMDB_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def read_single_lmdb(image_id, lmdb_dir="/data/lrudden/diffusion_TM/dataset/train_lmdb"):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """

    # Open the LMDB environment
    env = lmdb.open(lmdb_dir, readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        lmdb_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = lmdb_image.get_image()

    env.close()
    return image

def read_images(folder: str) -> np.array:
    directory = os.fsencode(folder)

    images = []
    for image_file_name in os.listdir(directory):
        filename = os.fsdecode(image_file_name)
        if filename.endswith(".png"):
            images.append(np.array(Image.open(filename)))
    return np.asarray(images).astype(np.uint8)[:,:,:,np.newaxis] # add on the channel dimension at the end (grayscale)

def read_labels(filename: str) -> np.array:
    return np.load(filename) # load the prestored labels from a numpy array

if __name__ == "__main__":
    folder="/data/lrudden/diffusion_TM/dataset"

    # MNIST EXAMPLE #
    #filename_images="train-images-idx3-ubyte"
    #filename_labels="train-labels-idx1-ubyte"
    #arr = idx2numpy.convert_from_file(filename_images)[:,:,:,np.newaxis] # add on the channel dimension at the end (grayscale)
    #labels = idx2numpy.convert_from_file(filename_labels)    # read the files
    #store_many_lmdb(arr, labels)

    # Diffuse TM database - must be run from inside correct folder
    arr = read_images(folder)
    labels = read_labels(folder + "/labels.npy")
    store_many_lmdb(arr, labels)
