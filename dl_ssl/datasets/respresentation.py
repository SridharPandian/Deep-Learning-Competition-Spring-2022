import torch
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from PIL import Image

from tqdm import tqdm

from dl_ssl.utils.image_transforms import basic_transform_function

class RepresentationImageDataset(Dataset):
    def __init__(self, image_size, data_path, unlabelled = True):
        # Loading images from the data path and sorting based on names
        images_list = os.listdir(data_path)
        images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if unlabelled is True:
            mean = torch.tensor([0.0, 0.0, 0.0]) # Mean for the unlabelled dataset
            std = torch.tensor([1.0, 1.0, 1.0])  # Std for the unlabelled dataset
        else:
            mean = torch.tensor([0.0, 0.0, 0.0]) # Mean for the labelled dataset
            std = torch.tensor([1.0, 1.0, 1.0])  # Std for the labelled dataset

        # Transformation function for the images
        image_preprocessor = basic_transform_function(image_size, mean, std)

        # Initializing the empty image tensor array to store the entire dataset
        self.image_tensors = []

        # Obtaining all the images from the directory
        for image_name in tqdm(images_list):
            # Getting the absolute image path
            image_path = os.path.join(data_path, image_name)

            # Processing the image
            try:
                image = Image.open(image_path)
                image_tensor = image_preprocessor(image)
                self.image_tensors.append(image_tensor.detach())
                image.close()
            except:
                print(f"Cannot read image: {image_name}")
                continue


    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        '''
        Return the image and a zero vector containing four elements (since this is unlabelled data)
        '''
        return (self.image_tensors[idx], torchtorch.zeros(4))