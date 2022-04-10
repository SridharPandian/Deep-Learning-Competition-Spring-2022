# Standard imports
import argparse

# Torch imports
import torchvision.transforms as T

# Dataset and Function imports
from dl_ssl.datasets.dataset import *
from dl_ssl.utils.normalize import get_normal_values

# Other utility imports
from dl_ssl.utils.files import get_path_in_package

parser = argparse.ArgumentParser()

parser.add_argument('--labelled', action='store_true')
parser.add_argument('--unlabelled', action='store_true')

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

if __name__ == "__main__":
    options = parser.parse_args()
    
    # Getting metrics for unlabelled data
    if options.unlabelled is True:    
        dataset = UnlabeledDataset(root = get_path_in_package('data/unlabeled_data'), img_size = 224)
        get_normal_values(dataset, 224, 224)

    # Getting metrics for labelled data
    if options.labelled is True:    
        dataset = UnlabeledDataset(root = get_path_in_package('data/labeled_data/training/images'), img_size = 224, unlabelled = False)
        get_normal_values(dataset, 224, 224)