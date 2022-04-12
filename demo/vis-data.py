import torch
from dataset import UnlabeledDataset, LabeledDataset
import transforms as T
import utils
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
    
#data_path = '/Users/snehasilwal/Downloads/labeled_data'
data_path='/home/ss14499/labeled_data'
train_dataset = LabeledDataset(root=data_path, split="training", transforms=get_transform(train=True))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)


for img, label  in train_dataset:
    #import pdb
    #pdb.set_trace()
    fig, ax = plt.subplots()
    ax.set_title(label)
    coords = label['boxes']
    print(coords)
    print(type(coords))
    #pdb.set_trace()
    
    ax.imshow(img.permute(1,2,0))
    for i in range(coords.shape[0]):
        rect = patches.Rectangle((coords[i][0],coords[i][1]), coords[i][2] - coords[i][0], coords[i][3] - coords[i][1],linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
plt.show()
