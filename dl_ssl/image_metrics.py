# Standard imports
import argparse

# Dataset and Function imports
from dl_ssl.datasets.respresentation import RepresentationImageDataset
from dl_ssl.utils.normalize import obtain_normal_values

parser = argparse.ArgumentParser()

parser.add_argument('--labelled', action='store_true')
parser.add_argument('--unlabelled', action='store_true')

if __name__ == "__main__":
    options = parser.parse_args()

    # Getting metrics for unlabelled data
    if options.labelled is True:    
        dataset = RepresentationImageDataset(224, '/home/sridhar/.personal/deep-learning/project/dl_ssl/data/unlabelled_data')
        obtain_normal_values(dataset, 224, 224)

    # Getting metrics for labelled data
    if options.labelled is True:    
        dataset = RepresentationImageDataset(224, '/home/sridhar/.personal/deep-learning/project/dl_ssl/data/labelled_data/training')
        obtain_normal_values(dataset, 224, 224)