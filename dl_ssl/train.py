# Standard imports
import os
import numpy as np

# Torch imports
import torch
from torch.utils.data import DataLoader

# Model imports
from byol_pytorch import BYOL
from torchvision import models

# Dataset imports
from dl_ssl.datasets.dataset import LabeledDataset, UnlabeledDataset

# Graphical imports
import wandb

# Miscellaneous imports
import argparse
from tqdm import tqdm

# Other utility imports
from dl_ssl.utils.transforms import augmentation_generator
from dl_ssl.utils.files import *

# Parser import
from dl_ssl.utils.parsers import TrainParser

CHKPT_PATH = get_path_in_package('checkpoints')


def train_byol(options):
    # Selecting the GPU
    device = torch.device(f"cuda:{options.gpu_num}")
    print(f"Using GPU: {options.gpu_num} for training the model.")

    # Loading the dataset
    dataset = UnlabeledDataset(root = options.train_data_path, img_size = options.img_size, unlabelled = not options.labelled)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = True, num_workers = 24, pin_memory = True)
   
    # Loading an encoder
    if options.encoder == "resnet18":
        encoder = models.resnet18(pretrained = False).to(device)
    if options.encoder == "resnet50":
        encoder = models.resnet50(pretrained = False).to(device)
    if options.encoder == "resnet34":
        encoder = models.resnet34(pretrained = False).to(device)
    if options.encoder == "vitbase16":
        encoder = x = models.vit_b_16(pretrained = False).to(device)
    if options.encoder == "vitbase32":
        encoder = x = models.vit_b_32(pretrained = False).to(device)

    # Creating a BYOL model
    if options.augment_imgs is True:
        augment_custom = augmentation_generator()
        model = BYOL(
            encoder,
            image_size = options.img_size,
            augment_fn = augment_custom,
            hidden_layer = 'avgpool'
        )
    else:
        model = BYOL(
            encoder,
            image_size = options.img_size,
            hidden_layer = 'avgpool'
        )

    # Initializing WandB project
    wandb.init(project = "Deep Learning - SSL")
    if options.labelled is True:
        wandb.run.name = f'BYOL - labelled - v{options.run} - {options.encoder}'
    else:
        wandb.run.name = f'BYOL - unlabelled - v{options.run} - {options.encoder}'

    wandb.config = {
        "learning_rate": options.lr,
        "epochs": options.epochs,
        "batch_size": options.batch_size
    }

    # Initializing the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = options.lr)

    # Starting the training loop
    print("Starting the training process!")
    low_train_loss = torch.inf

    # Creating a checkpoint directory
    if options.labelled is False:
        run_chkpt_dir = os.path.join(CHKPT_PATH, f'byol_unlabelled_run_{options.run}_{options.encoder}')
    else:
        run_chkpt_dir = os.path.join(CHKPT_PATH, f'byol_labelled_run_{options.run}_{options.encoder}')

    make_dir(run_chkpt_dir)

    for epoch in range(options.epochs):
        train_loss = 0

        for images in tqdm(dataloader):
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

def train_moco(options, device):
    pass
    # Selecting the GPU
    # device = torch.device(f"cuda:{options.gpu_num}")
    # print(f"Using GPU: {options.gpu_num} for training the model.")

    # # Loading the dataset
    # dataset = UnlabeledDataset(root = options.train_data_path, img_size = options.img_size, unlabelled = not options.labelled)
    # dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = True, num_workers = 24, pin_memory = True)
   
    # # Initializing the model
    # # TODO

    # # Initializing WandB project
    # wandb.init(project = "Deep Learning - SSL")
    # wandb.run.name = f'BYOL - v{options.run}'
    # wandb.config = {
    #     "learning_rate": options.lr,
    #     "epochs": options.epochs,
    #     "batch_size": options.batch_size
    # }

    # # Initializing WandB project
    # wandb.init(project = "Deep Learning - SSL")
    # if options.labelled is True:
    #     wandb.run.name = f'MoCo - labelled - v{options.run}'
    # else:
    #     wandb.run.name = f'MoCo - unlabelled - v{options.run}'
        
    # wandb.config = {
    #     "learning_rate": options.lr,
    #     "epochs": options.epochs,
    #     "batch_size": options.batch_size
    # }

    # # Initializing the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr = options.lr)

    # # Starting the training loop
    # print("Starting the training process!")
    # low_train_loss = torch.inf

    # # Creating a checkpoint directory
    # if options.labelled is False:
    #     run_chkpt_dir = os.path.join(CHKPT_PATH, f'moco_unlabelled_run_{options.run}')
    # else:
    #     run_chkpt_dir = os.path.join(CHKPT_PATH, f'moco_labelled_run_{options.run}')

    # make_dir(run_chkpt_dir)

    # for epoch in range(options.epochs):
    #     train_loss = 0

    #     for images in tqdm(dataloader):
    #         optimizer.zero_grad()

    #         # TODO
    #         # loss = model(images.float().to(device))

    #         loss.backward()
    #         optimizer.step()

    #         train_loss += loss.item() * images.shape[0]

    #     # Logging the losses
    #     print(f"Training Loss encountered: {train_loss / len(dataset)}")
    #     wandb.log({'train_loss': train_loss / len(dataset)})

    #     if epoch % 20 == 0:
    #         epoch_checkpt_path = os.path.join(run_chkpt_dir, f"checkpoint-{epoch + 1}.pth")
    #         torch.save({'model_state_dict': encoder.state_dict()}, epoch_checkpt_path)
    #     if train_loss / len(dataset) < low_train_loss:
    #         epoch_checkpt_path = os.path.join(run_chkpt_dir, f"checkpoint-lowest-loss.pth")
    #         torch.save({'model_state_dict': encoder.state_dict()}, epoch_checkpt_path)

if __name__ == "__main__":
    # Obtaining the input arguments
    parser = TrainParser()
    options = parser.parse()

    # Training based on the method we need
    if options.model == "byol":
        train_byol(options)
    if options.model == "moco":
        train_moco(options)