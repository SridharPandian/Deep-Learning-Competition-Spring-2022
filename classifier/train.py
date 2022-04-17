#TODO load BYOL weights, eval mode
#run through classifier, compare to labeled data outputs
# Standard imports
import os
import numpy as np

# Torch imports
import torch
from torch.utils.data import DataLoader
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
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


rpn_network_filepath='../demo/train_04_10_2022_18_57_15/_epoch3.tar'
filepath='/data/sridhar/checkpoints/byol_unlabelled_run_4/checkpoint-101.pth'


#TODO probably can do the following importing of pre-trained models more neatly
def create_byol_model(device, chkpt_weights, augment_img=True):
    # Creating the BYOL model
    encoder = models.resnet50(pretrained = False).to(device)

    encoder.load_state_dict(chkpt_weights)
    encoder.eval()
    return encoder

    # if augment_img is True:
    #     augment_custom = augmentation_generator()
    #     model = BYOL(
    #         encoder,
    #         image_size = options.img_size,
    #         augment_fn = augment_custom
    #     )
    # else:
    #     model = BYOL(
    #         encoder,
    #         image_size = options.img_size
    #     )

def get_rpn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def create_rpn_network(device, rpn_weights):
    model = get_rpn_model(100).to(device)
    model.load_state_dict(rpn_weights)
    model.eval()
    return model


def train_model():
    # Selecting the GPU
    device = torch.device(f"cuda:{options.gpu_num}")
    print(f"Using GPU: {options.gpu_num} for training the model.")
    

    num_classes = 100
    train_dataset = LabeledDataset(root=labeled_data_path, split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root=labeled_data_path, split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)


    rpn = get_rpn_model(device, rpn_network_filepath)
    enc = create_byol_model(device,filepath)

    classifier = SimpleClassifier(device, enc)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

