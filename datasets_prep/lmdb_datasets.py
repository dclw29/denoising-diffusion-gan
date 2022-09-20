# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image
import pickle
import torchvision.transforms as transforms

def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == "diffusion_TM":
        return 42492 if train else 2000 # this number is specific to the number of TM examples we have
    elif dataset == "mnist":
        return 60000 if train else 4000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)

class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root)
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):

        with self.data_lmdb.begin(write=False, buffers=True) as txn:

            # LSPR changes
            #data = txn.get(str(index).encode())
            data = txn.get(f"{index:08}".encode())

            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                #img = img.convert('L') # L from RGB to deal with grayscale
                img = img.convert('RGB')
            else:
                #### Original ###
                #img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                #size = int(np.sqrt(len(img) / 3))
                #img = np.reshape(img, (size, size, 3))
                #img = Image.fromarray(img, mode='RGB')

                ### LSPR ###
                lmdb_image = pickle.loads(data)
                # assume it is grayscale
                #size = int(np.sqrt(len(img) / 1))
                #img = np.reshape(img, (size, size, 1))
                #img = Image.fromarray(img, mode='L')
                img, label = lmdb_image.get_image()
                print(np.shape(img))

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return num_samples(self.name, self.train)

