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

def train_one_epoch((rpn,enc,model), optimizer, data_loader, device, epoch, print_freq):
    rpn.eval()
    enc.eval()
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        import pdb
        pdb.set_trace()

        #first get regions of interest, pass through pre-trained labeled region proposal
        regions = rpn(images,targets)['boxes']
        #TODO some function to get boxes

        #pass proposed regions of interest through SSL method, to get embeddings
        embeddings = enc(regions)

        loss_dict = model(embeddings, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def train_model(num_epochs=5):
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

    classifier = SimpleClassifier(enc).to(device)
    classifier.train()

    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for e in num_epochs:
        train_epoch((rpn,enc,classifier), optimizer,train_loader, device, e, 10)
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)



