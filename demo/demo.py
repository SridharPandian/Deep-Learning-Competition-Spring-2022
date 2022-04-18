# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
import transforms as T
import utils
from engine import train_one_epoch, evaluate
from datetime import datetime

from dataset import UnlabeledDataset, LabeledDataset

byol_weights_filepath='/data/sridhar/checkpoints/byol_unlabelled_run_4/checkpoint-101.pth'
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

def get_fasterRCNN(num_classes = 100):
    ssl_model = torchvision.models.resnet50(pretrained=False)
    b = torch.load(open(byol_weights_filepath,'rb'))
    ssl_model.load_state_dict(b['model_state_dict'])

    #ssl_bb = torch.nn.Sequential(*(list(torchvision.models.resnet50(pretrained=False).children())[:-2]))

    ssl_bb = torch.nn.Sequential(*(list(ssl_model.children())[:-2]))
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # model.backbone.body = ssl_bb
    ssl_bb.out_channels = 2048
    # #TODO Load SSL trained weights here
    # from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    # backbone = resnet_fpn_backbone('resnet50', False, trainable_layers=3)
    # # This just adds the fpn to the resnet backbone?
    # backbone.body = ssl_bb
    anchor_sizes = ((512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    default_anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = torchvision.models.detection.FasterRCNN(backbone = ssl_bb, num_classes=100, rpn_anchor_generator=default_anchor_gen)   
    # # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    now = datetime.now()
    job_dir = now.strftime("train_%m_%d_%Y_%H_%M_%S")
    os.mkdir(job_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 100
    train_dataset = LabeledDataset(root='../../../labeled_data', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='../../../labeled_data', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    # model = get_model(num_classes)
    model = get_fasterRCNN(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)
        if(epoch%5 == 0): 
            # save model weights
            output_file = job_dir + '/_epoch' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, output_file)

    print("That's it!")

if __name__ == "__main__":
    main()
