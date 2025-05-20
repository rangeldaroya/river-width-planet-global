from torch.utils.data.dataset import Dataset

import rasterio
import os
import torch
import fnmatch
import numpy as np
import pandas as pd
import pdb
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F
from loguru import logger

# import h5py
# import json
# import pickle
# import ast
from skimage.transform import rescale, resize

# Function to downsample by averaging over blocks of pixels
def downsample_image(image, factor):
    # Rescale the image to the lower resolution using resize
    downsampled_image = resize(image, 
                               (image.shape[0] // factor, image.shape[1] // factor, image.shape[2]), 
                               order=1,     # bilinear interpolation
                               mode='reflect', 
                               anti_aliasing=True)
    return downsampled_image

# Function to upsample by interpolation
def upsample_image(image, target_shape):
    # Resize the image back to the original resolution (3m/pixel)
    upsampled_image = resize(image, target_shape, order=1, mode='reflect', anti_aliasing=True)  # bilinear interpolation
    return upsampled_image

class PlanetSegmentation(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(
        self, 
        return_fp = False,
        resize_size=None,
        is_downsample=False,    # Set to true to downsample image resolution
    ):
        self.input_col_name = "tile_fp"  # tif

        self.num_channels = 4   # RGB+NIR
        self.return_fp = return_fp
        
        
        all_fns = pd.read_csv("all_tiles_planet.csv")
        self.fns = all_fns

        self.data_len = len(self.fns)
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        self.transforms_list = [transforms.ToTensor()]
        self.transforms_list += [transforms.Resize(size=(final_size,final_size))]
        # NOTE: if adding more transforms, make sure to apply same transforms to label!!

        self.is_downsample = is_downsample
        
        logger.debug(f"self.is_downsample: {self.is_downsample}")
        # if split == "train":
        #     self.data_len = 45

    def __getitem__(self, index):
        data_row = self.fns.iloc[index]
        input_fp = data_row[self.input_col_name]
        # label_fp = data_row[self.label_col_name]
        # logger.debug(f"input_fp: {input_fp}")
        
        image = rasterio.open(input_fp).read()
        image = np.transpose(image, (1,2,0))    # (500,500,4)

        label = np.zeros_like(image[:,:,:1])
        
        data_transforms = transforms.Compose(self.transforms_list)
        data = np.concatenate((image,label), axis=-1)
        trans_data = data_transforms(data)  # output of transforms: (5, 512, 512)
        image = trans_data[:self.num_channels, :, :].float()
        label = trans_data[-1, :, :].float()

        # Min-max Normalization
        # Normalize data
        # """
        if (torch.max(image)-torch.min(image)):
            image = image - torch.min(image)
            image = image / torch.maximum(torch.max(image),torch.tensor(1))
        else:
            # logger.warning(f"all zero image. setting all labels to zero. index: {index}. {input_fp}")
            image = torch.zeros_like(image).float()
            label = torch.zeros_like(label).float()
            # all_data = torch.zeros_like(all_data)
        # """
        # logger.debug(f"label max: {torch.max(label)}. min: {torch.min(label)}. unique: {torch.unique(label)}")
        # logger.debug(f"label: {label.shape}")   # (512,512)
        # logger.debug(f"image: {image.shape}")   # (4, 512, 512)
        labels = {
            "water_mask": label.float(),
        }

        if self.return_fp:
            return (
                image,    # shape: (3, 512, 512)
                labels,
                input_fp,
            )

        else:
            return (
                image,    # shape: (3, 512, 512)
                labels,
            )


    def __len__(self):
        return self.data_len

