# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader

# Status bar
from tqdm import tqdm

def get_normal_values(dataset, image_height, image_width):
    print(f"Number of images in the dataset: {len(dataset)}")

    # Creating the dataloader
    dataloader = DataLoader(dataset, batch_size = 128, shuffle = False, num_workers = 24)
    print("Loaded")
    # Initializing tensors
    total = torch.tensor([0.0, 0.0, 0.0])
    sq_total = torch.tensor([0.0, 0.0, 0.0])

    for images, targets in tqdm(dataloader):
        
        total += images.sum(axis = [0, 2, 3])
        sq_total += (images ** 2).sum(axis = [0, 2, 3])

    total_num_vals = len(dataset) * image_height * image_width

    mean = total / count
    variance = (sq_total / count) - (mean ** 2)
    std = torch.sqrt(variance)

    print(f"Mean of the dataset: {mean}")
    print(f"STD of the dataset: {std}")