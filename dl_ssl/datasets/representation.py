# Standard imports
import os
import numpy as np

# Torch based imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# Image imports
from PIL import Image

# Status bar import
from tqdm import tqdm

# Transformation function
from dl_ssl.utils.image_transforms import basic_transform_function

class RepresentationImageDataset(Dataset):
    def __init__(self, image_size, data_path, unlabelled = True):
        # Loading images from the data path and sorting based on names
        self.base_path = data_path
        self.images_list = os.listdir(data_path)
        self.images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if unlabelled is True:
            mean = torch.tensor([0.0, 0.0, 0.0]) # Mean for the unlabelled dataset
            std = torch.tensor([1.0, 1.0, 1.0])  # Std for the unlabelled dataset
        else:
            mean = torch.tensor([0.0, 0.0, 0.0]) # Mean for the labelled dataset
            std = torch.tensor([1.0, 1.0, 1.0])  # Std for the labelled dataset

        # Transformation function for the images
        self.image_preprocessor = basic_transform_function(image_size, mean, std)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        '''
        Return the image and a zero vector containing four elements (since this is unlabelled data)
        '''
        # Obtaining the image[idx] path
        image_path = os.path.join(self.base_path, self.images_list[idx])

        # Processing and outputing the image
        with open(image_path, 'rb') as file:
            image = Image.open(file)
            image_tensor = self.image_preprocessor(image)

        return (image_tensor, torch.zeros(4))