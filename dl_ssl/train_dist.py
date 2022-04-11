# Standard imports
import os
import numpy as np

# Torch imports
import torch
from torch.utils.data import DataLoader

# Distributed imports
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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

# Setup the Process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank = rank, world_size = world_size)

# Clean all the process groups
def cleanup():
    dist.destroy_process_group()

# Prepare the distributed dataloader
def prepare_dist_data(dataset, rank, world_size, batch_size, pin_memory, num_workers):
    sampler = DistributedSampler(dataset, num_replicas = world_size, rank = rank, shuffle = False, drop_last = False)
    dataloader = DataLoader(dataset, batch_size = batch_size, pin_memory = pin_memory, num_workers = num_workers, drop_last = False, shuffle = False, sampler = sampler)

    return dataloader

def train_byol(rank, world_size, options):
    # Setting up the process groups
    setup(rank, world_size)

    # Loading the dataset
    dataset = UnlabeledDataset(root = options.train_data_path, img_size = options.img_size, unlabelled = not options.labelled)
    dataloader = prepare_dist_data(
        dataset = dataset,
        rank = rank,
        world_size = world_size,
        batch_size = options.batch_size,
        pin_memory = False,
        num_workers = 0
    )

    # Creating the model
    encoder = models.resnet50(pretrained = False).to(rank)
    encoder = DDP(encoder, device_ids = [rank], output_device = rank, find_unused_parameters = True)
    if options.augment_imgs is True:
        augment_custom = augmentation_generator()
        learner = BYOL(
            encoder,
            image_size = options.img_size,
            augment_fn = augment_custom
        )
    else:
        learner = BYOL(
            encoder,
            image_size = options.img_size
        )

    # Initializing WandB project
    wandb.init(project = "Deep Learning - SSL")
    wandb.run.name = f'BYOL - v{options.run}'
    wandb.config = {
        "learning_rate": options.lr,
        "epochs": options.epochs,
        "batch_size": options.batch_size
    }

    # Initializing the optimizer
    optimizer = torch.optim.Adam(learner.parameters(), lr = options.lr)

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

        # Announcing the epoch number to the DistributedSampler
        dataloader.sampler.set_epoch(epoch)

        for images in tqdm(dataloader):
            optimizer.zero_grad()

            loss = model(images.float().to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.shape[0]

        # Logging the losses
        print(f"Training Loss encountered: {train_loss / len(dataset)}")
        wandb.log({'train_loss': train_loss / len(dataset)})
        wandb.watch(encoder)

        if epoch % 20 == 0:
            epoch_checkpt_path = os.path.join(run_chkpt_dir, f"checkpoint-{epoch + 1}.pth")
            torch.save({'model_state_dict': encoder.state_dict()}, epoch_checkpt_path)
        if train_loss / len(dataset) < low_train_loss:
            epoch_checkpt_path = os.path.join(run_chkpt_dir, f"checkpoint-lowest-loss.pth")
            torch.save({'model_state_dict': encoder.state_dict()}, epoch_checkpt_path)

    cleanup()

if __name__ == '__main__':
    # Obtaning the input arguments
    parser = TrainParser()
    options = parser.parse()

    world_size = options.num_gpus

    # Training based on the method we need
    if options.model == "byol":
        mp.spawn(
            train_byol,
            args = (world_size, options),
            nprocs = world_size
        )