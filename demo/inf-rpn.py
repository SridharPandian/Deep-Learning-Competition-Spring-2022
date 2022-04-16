import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import UnlabeledDataset, LabeledDataset
import transforms as T
import utils
from matplotlib import pyplot as plt
import matplotlib.patches as patches

checkpoint = torch.load('train_04_10_2022_17_15_51_epoch0.tar')

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

num_classes = 100
model = get_model(num_classes)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval()

data_path='/home/ss14499/labeled_data'
train_dataset = LabeledDataset(root=data_path, split="training", transforms=get_transform(train=True))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

for img, label  in train_dataset:
    res = model([img])[0]
    fig, ax = plt.subplots()
    ax.set_title(label)
    
    coords = res['boxes'].detach()[:5].numpy()
    print(coords)
    #ax.imshow(img.permute(1,2,0))
    for i in range(coords.shape[0]):
        rect = patches.Rectangle((coords[i][0],coords[i][1]), coords[i][2] - coords[i][0], coords[i][3] - coords[i][1],linewidth=1, edgecolor='r', facecolor='none')
        #ax.add_patch(rect)
        import pdb
        pdb.set_trace()
        cropped_img = torchvision.transforms.functional.crop(img,int(coords[i][1]),int(coords[i][0]),int( coords[i][3] - coords[i][1]), int(coords[i][2] - coords[i][0]))
         
        plt.imshow(cropped_img.permute(2,1,0))
        print(cropped_img)
        plt.show()
        
    plt.show()
    import pdb
    pdb.set_trace()
