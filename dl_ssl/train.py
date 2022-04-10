# Standard imports
import os
import numpy as np

# Torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import models

# Dataset imports
from dl_ssl.datasets.representation import RepresentationImageDataset

# Graphical imports
import wandb

# Miscellaneous imports
import argparse
from tqdm import tqdm

# Other utility imports
from dl_ssl.utils.image_transforms import augmentation_generator
from dl_ssl.utils.files import *

# Parser import
from dl_ssl.utils.parsers import TrainParser

CHKPT_PATH = get_path_in_package('checkpoints')


def train_byol(options, device):
    # Loading the dataset
    dataset = RepresentationImageDataset(options.img_size, options.train_data_path, options.unlabelled)

    # Creating the dataloader
    dataloader = DataLoader(dataset, options.batch_size, shuffle = True, pin_memory = True, num_workers = 24)
    
    # Creating the BYOL model
    encoder = models.resnet50(pretrained = False).to(device)
    if options.augment_imgs is True:
        augment_custom = augmentation_generator()
        model = BYOL(
            encoder,
            image_size = options.img_size,
            augment_fn = augment_custom
        )
    else:
        model = BYOL(
            encoder,
            image_size = options.img_size
        )

    # Initializing WandB project
    wandb.init(project = "Deep Learning - SSL")

    # Initializing the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = options.lr)

    # Starting the training loop
    print("Starting the training process!")
    low_train_loss = torch.inf

    # Creating a checkpoint directory
    if options.labelled is False:
        run_chkpt_dir = os.path.join(CHKPT_PATH, f'byol_labelled_run_{options.run}')
    else:
        run_chkpt_dir = os.path.join(CHKPT_PATH, f'byol_unlabelled_run_{options.run}')

    for epoch in range(options.epochs):
        train_loss = 0

        for images, targets in tqdm(dataloader):
            optimizer.zero_grad()

            loss = model(images.float().to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.shape[0]

        # Logging the losses
        print(f"Training Loss encountered: {train_loss / len(dataset)}")
        wandb.log({'train_loss': train_loss / len(dataset)})

        if epoch % 20 == 0:
            epoch_checkpt_path = os.path.join(run_chkpt_dir, f"checkpoint-{epoch + 1}.pth")
            torch.save({'model_state_dict': encoder.state_dict()}, epoch_checkpt_path)
        if train_loss / len(dataset) < low_train_loss:
            epoch_checkpt_path = os.path.join(run_chkpt_dir, f"checkpoint-lowest-loss.pth")
            torch.save({'model_state_dict': encoder.state_dict()}, epoch_checkpt_path)

if __name__ == "__main__":
    # Obtaining the input arguments
    parser = TrainParser()
    options = parser.parse()

    # Selecting the GPU
    device = torch.device(f"cuda:{options.gpu_num}")
    print(f"Using GPU: {options.gpu_num} for training the model.")

    # Training based on the method we need
    if options.model == "byol":
        train_byol(options, device)