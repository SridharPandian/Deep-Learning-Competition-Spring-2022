# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

import os
import sys
sys.path.append(os.getcwd() + '/../demo')
import transforms as T
import utils
from engine import train_one_epoch, evaluate
import wandb
import datetime
import argparse

from dataset import UnlabeledDataset, LabeledDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_fasterRCNN(device, backbone_type, backbone_path, num_classes = 100):
    ssl_model = torchvision.models.resnet50(pretrained=False)
    ssl_model.fc = nn.Identity()
    ssl_weights = None
    if(backbone_type =='dino'):
        ssl_weights = load_dino_weights(device, checkpoint_location=backbone_path)
    elif(backbone_type=='moco'): 
        ssl_weights = load_moco_weights(device,checkpoint_location=backbone_path)
    else:
        print('Backbone type not supported, sorry')
        sys.exit()

    ssl_model.load_state_dict(ssl_weights, strict = False)
    ssl_bb = torch.nn.Sequential(*(list(ssl_model.children())[:-1]))

    ssl_bb.out_channels = 2048

    anchor_sizes = ((512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    default_anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
    model = torchvision.models.detection.FasterRCNN(backbone = ssl_bb, num_classes=100, rpn_anchor_generator=default_anchor_gen)   
    # # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def load_dino_weights(device, checkpoint_location):
    state_dict = torch.load(checkpoint_location,map_location=device)["student"]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict

def load_moco_weights(device,checkpoint_location):
    state_dict = torch.load(checkpoint_location)["state_dict"]
    state_dict = {k.replace("module.base_encoder.", ""): v for k, v in state_dict.items()}
    return state_dict


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    exp_name = args.exp_name + str(datetime.datetime.now())
    
    wandb.init(
            project="dl-project-finetune", 
            name=exp_name,
    )
    

    num_classes = 100
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

    # model = get_model(num_classes)
    model = get_fasterRCNN(device, args.backbone_type, args.backbone_path, num_classes)
    if(args.restart_from):
        print('Resuming from checkpoint: ' + args.restart_from + '... ')
        model.load_state_dict(torch.load(args.restart_from,map_location=device))

    model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.03, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 100
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)
        print("Saving model")
        output_path = os.path.join(args.output_dir, 'finetuned{}'.format(epoch))
        torch.save(model.state_dict(), output_path)


    print("That's it!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify details about model training--output and files to use.')
    parser.add_argument('--backbone_path', type=str,required=True)
    parser.add_argument('-o','--output_dir', type=str,required=True)
    parser.add_argument('--backbone_type', type=str,default="dino",required=True)
    parser.add_argument('-e','--exp_name', type=str,required=True)
    parser.add_argument('-r','--restart_from',type=str)
    args = parser.parse_args()
    main(args)
