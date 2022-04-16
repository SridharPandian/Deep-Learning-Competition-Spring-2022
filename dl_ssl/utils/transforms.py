import torch
from torchvision import transforms as T

def basic_transform(image_size, unlabelled):
    # Metrics for the unlabelled dataset
    if unlabelled:
        norm_values = T.Normalize(
            mean = torch.tensor([0.4917, 0.4694, 0.4148]), 
            std = torch.tensor([0.2856, 0.2782, 0.2981])  
        )
    # Metrics for the labelled dataset
    else:
        norm_values = T.Normalize(
            mean = torch.tensor([0.4699, 0.4516, 0.3953]), 
            std = torch.tensor([0.2719, 0.2645, 0.2762])  
        )

    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size, image_size)),
        norm_values
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def augmentation_generator(unlabelled = True):
    return T.Compose([
        T.RandomResizedCrop(224, scale = (0.3, 1)),  
        T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
        T.RandomGrayscale(p = 0.2),
        T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
        T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]))
    ])
