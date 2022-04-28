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
import wandb
import datetime
import argparse

from dataset import UnlabeledDataset, LabeledDataset

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ssl_method', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--data_path', default='/home/robotlab/projects/deep-learning/labeled_data' ,type=str)
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    return parser.parse_args()

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

def get_fasterRCNN(args, num_classes = 100):
    ssl_model = torchvision.models.resnet50(pretrained=False)
    ssl_model.fc = nn.Identity()

    if args.ssl_method == 'dino':
        ssl_model.load_state_dict(load_dino_weights(checkpoint_location=args.checkpoint_dir), strict = False)
    elif args.ssl_method == 'byol':
        ssl_model.load_state_dict(load_byol_weights(checkpoint_location=args.checkpoint_dir), strict = False)
    elif args.ssl_method == 'moco':
        ssl_model.load_state_dict(load_moco_weights(checkpoint_location=args.checkpoint_dir), strict = False)

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

def load_dino_weights(checkpoint_location):
    state_dict = torch.load(checkpoint_location)["student"]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict

def load_moco_weights(checkpoint_location):
    state_dict = torch.load(checkpoint_location)["state_dict"]
    state_dict = {k.replace("module.base_encoder.", ""): v for k, v in state_dict.items()}
    return state_dict

def load_byol_weights(checkpoint_location):
    state_dict = torch.load(checkpoint_location)["model_state_dict"]
    return state_dict

def main(args):
    print(f'Using GPU: {args.gpu_num} for training!')
    device = torch.device(f'cuda:{args.gpu_num}') if torch.cuda.is_available() else torch.device('cpu')
    if args.ssl_method == 'dino':
        exp_name = "finetune_dino_"+ str(args.run)
    elif args.ssl_method == 'byol':
        exp_name = "finetune_byol_"+ str(args.run)
    elif args.ssl_method == 'moco':
        exp_name = "finetune_moco_"+ str(args.run)
    
    wandb.init(
            project="dl-project-finetune", 
            name=exp_name,
    )
    

    num_classes = 100
    train_dataset = LabeledDataset(root=args.data_path, split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root=args.data_path, split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=utils.collate_fn)

    # model = get_model(num_classes)
    model = get_fasterRCNN(args, num_classes)
    model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)
        print("Saving model")

        if exp == 'dino':
            torch.save(model.state_dict(), "/home/robotlab/projects/deep-learning/Deep-Learning-Competition-Spring-2022/demo/checkpoints/dino_finetuned_{}.pth".format(epoch))
        elif exp == 'byol':
            torch.save(model.state_dict(), "/home/robotlab/projects/deep-learning/Deep-Learning-Competition-Spring-2022/demo/checkpoints/byol_finetuned_{}.pth".format(epoch))
        elif exp == 'moco':
            torch.save(model.state_dict(), "/home/robotlab/projects/deep-learning/Deep-Learning-Competition-Spring-2022/demo/checkpoints/moco_finetuned_{}.pth".format(epoch))


    print("That's it!")

if __name__ == "__main__":
    options = get_args()
    main(options)
