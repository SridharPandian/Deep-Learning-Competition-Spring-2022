# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
import transforms as T
import utils
from engine import train_one_epoch, evaluate
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

def get_fasterRCNN(bbpath,mpath,num_classes=100):
    ssl_model = torchvision.models.resnet50(pretrained=False)
    ssl_model.fc = nn.Identity()
    ssl_model.load_state_dict(load_dino_weights(checkpoint_location=bbpath), strict = False)

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
    model.load_state_dict(torch.load(mpath)())
    return model

def load_dino_weights(checkpoint_location):
    state_dict = torch.load(checkpoint_location)["student"]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict



def main():
    parser = argparse.ArgumentParser(description='Specify model files to evaluate.') 
    parser.add_argument('-b', '--bb_path', default="./checkpoint0033.pth", type=str)
    parser.add_argument('-m', '--model_path', default="./finetuned_0.pth", type=str)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    num_classes = 100
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=5, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    # model = get_model(num_classes)
    model = get_fasterRCNN(args.bb_path, args.model_path,num_classes)
    model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    evaluate(model, valid_loader, device=device)


    print("That's it!")

if __name__ == "__main__":
    main()
